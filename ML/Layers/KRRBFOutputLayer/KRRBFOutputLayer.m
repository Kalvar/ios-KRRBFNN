//
//  KRRBFOutputLayer.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFOutputLayer.h"
#import "KRRBFPattern.h"
#import "KRRBFCenterNet.h"
#import "KRRBFOutputNet.h"
#import "KRRBFActiviation.h"

@interface KRRBFOutputLayer()

@property (nonatomic, strong) KRRBFActiviation *activeFunction;

@end

@implementation KRRBFOutputLayer (OutputCalculation)

-(double)_calculateMaxDistanceWithCenters:(NSArray <KRRBFCenterNet *> *)_centers
{
    double _maxDistance = -1.0f;
    for( KRRBFCenterNet *_center in _centers )
    {
        for( KRRBFCenterNet *_anotherCenter in _centers )
        {
            double _distance = [self.activeFunction euclidean:_center.features x2:_anotherCenter.features];
            if( _distance > _maxDistance )
            {
                _maxDistance = _distance;
            }
        }
    }
    return _maxDistance;
}

// To calculate that common sigma of centers (all centers use the same sigma).
-(double)_calculateCommonSigmaWithCenters:(NSArray <KRRBFCenterNet *> *)_centers
{
    double _maxDistance = [self _calculateMaxDistanceWithCenters:_centers];
    return (_maxDistance >= 0.0f) ? _maxDistance / sqrt([_centers count]) : 0.0f;
}

@end

@implementation KRRBFOutputLayer

+(instancetype)sharedLayer
{
    static dispatch_once_t pred;
    static KRRBFOutputLayer *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFOutputLayer alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _activeFunction = [KRRBFActiviation sharedActiviation];
        _nets           = [NSMutableArray new];
        _rmse           = 0.0f;
        _costError      = 0.0f;
    }
    return self;
}

-(void)removeAllNets
{
    if( nil != _nets )
    {
        [_nets removeAllObjects];
    }
}

-(void)addNetsFromArray:(NSArray<KRRBFOutputNet *> *)_outputNets
{
    [self removeAllNets];
    [_nets addObjectsFromArray:_outputNets];
}

#pragma --mark Setup Sigma
// 手動設定所有 Centers 共用的 Sigma
// LMS 學習法適合用 Common Sigma 的模式
// Through memory reference of centers to set their common sigma value in these centers.
-(void)setupCommonSigmaWithCenters:(NSArray<KRRBFCenterNet *> *)_centers
{
    double _commonSigma = [self _calculateCommonSigmaWithCenters:_centers];
    for( KRRBFCenterNet *_centerNet in _centers )
    {
        _centerNet.sigma = _commonSigma;
    }
}

// 手動設定所有 Centers 各自專屬的 Sigma
// matchSigmas is matching with centers that index by index.
-(void)setupSigmasWithCenters:(NSArray<KRRBFCenterNet *> *)_centers matchSigmas:(NSArray <NSNumber *> *)_matchSigmas
{
    NSInteger _index = -1;
    for( KRRBFCenterNet *_centerNet in _centers )
    {
        _index           += 1;
        _centerNet.sigma  = [[_matchSigmas objectAtIndex:_index] doubleValue];
    }
}

#pragma --mark Output Methods
// Since last training step is outputing of whole network,
// those centers of network must be set up their sigmas (fixed common sigma or 1 center has 1 sigma).
-(void)outputWithPatterns:(NSArray<KRRBFPattern *> *)_patterns centers:(NSArray<KRRBFCenterNet *> *)_centers completion:(KRRBFOutputLayerCompletion)_completion patternOutput:(KRRBFOutputLayerPatternOutput)_patternOutput
{
    _costError = 0.0f;
    for ( KRRBFPattern *_pattern in _patterns )
    {
        NSMutableArray *_rbfValues = [NSMutableArray new];
        for( KRRBFCenterNet *_center in _centers )
        {
            double _rbfValue = [self.activeFunction rbf:_pattern.features x2:_center.features sigma:_center.sigma];
            [_rbfValues addObject:[NSNumber numberWithDouble:_rbfValue]];
            // To record this RBF value with current center.
            _center.rbfValue = _rbfValue;
        }
        
        // Centers outputs to Network output nets, the output1, output2, ... outputN.
        // Recording outputs of each pattern to transfer for outside block usage.
        double _patternCost    = 0.0f;
        NSInteger _outputIndex = -1;
        for( KRRBFOutputNet *_outputNet in _nets )
        {
            _outputIndex            += 1;
            // To calculate network output value
            [_outputNet outputWithRBFValues:_rbfValues];
            // The target-output of pattern
            NSNumber *_patternTarget = [_pattern.targets objectAtIndex:_outputIndex];
            _outputNet.targetValue   = [_patternTarget doubleValue];
            // Cost value (error value) of pattern
            _patternCost            += _outputNet.costError;
        }
        
        // Cost value is summed from all cost values of patterns.
        _costError += _patternCost;
        
        if( _patternOutput )
        {
            _patternOutput(_nets, _patternCost, _pattern);
        }
    }
    
    // RMSE (MSE is without sqrt())
    _rmse = sqrt(_costError / ( [_patterns count] * [_nets count] ));
    if( _completion )
    {
        _completion(self);
    }
}

#pragma --mark Predication Methods
-(void)predicateWithPatterns:(NSArray<KRRBFPattern *> *)_patterns centers:(NSArray<KRRBFCenterNet *> *)_centers outputs:(KRRBFOutputLayerPredication)_outputsBlock
{
    NSMutableDictionary *_predications = [NSMutableDictionary new];
    for ( KRRBFPattern *_pattern in _patterns )
    {
        NSMutableArray *_rbfValues = [NSMutableArray new];
        for( KRRBFCenterNet *_center in _centers )
        {
            double _rbfValue = [self.activeFunction rbf:_pattern.features x2:_center.features sigma:_center.sigma];
            [_rbfValues addObject:[NSNumber numberWithDouble:_rbfValue]];
        }
        
        // Predicated outputs of each pattern.
        NSMutableArray *_outputs = [NSMutableArray new];
        for( KRRBFOutputNet *_outputNet in _nets )
        {
            [_outputNet outputWithRBFValues:_rbfValues];
            [_outputs addObject:[NSNumber numberWithDouble:_outputNet.outputValue]];
        }
        
        // Uses pattern.indexKey to record its outputs of predication.
        [_predications setObject:_outputs forKey:_pattern.indexKey];
    }
    
    if( _outputsBlock )
    {
        _outputsBlock(_predications);
    }
}

-(void)dealloc
{
    NSLog(@"KRRBFOutputLayer is dealloced");
}

@end
