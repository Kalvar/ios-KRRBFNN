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

-(double)_calculateSigmaWithCenters:(NSArray <KRRBFCenterNet *> *)_centers
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

-(void)outputWithPatterns:(NSArray <KRRBFPattern *> *)_patterns centers:(NSArray <KRRBFCenterNet *> *)_centers completion:(KRRBFOutputLayerCompletion)_completion eachOutput:(KRRBFOutputLayerOutput)_eachOutput
{
    double _sigma      = [self _calculateSigmaWithCenters:_centers];
    double _errorValue = 0.0f;
    for ( KRRBFPattern *_pattern in _patterns )
    {
        NSMutableArray *_rbfValues = [NSMutableArray new];
        for( KRRBFCenterNet *_center in _centers )
        {
            double _rbfValue = [self.activeFunction rbf:_pattern.features x2:_center.features sigma:_sigma];
            [_rbfValues addObject:[NSNumber numberWithDouble:_rbfValue]];
        }
        
        // Centers outputs to Network output nets, the output1, output2, ... outputN
        NSInteger _outputIndex = -1;
        for( KRRBFOutputNet *_outputNet in _nets )
        {
            _outputIndex            += 1;
            // Network output value
            [_outputNet outputWithRBFValues:_rbfValues];
            // The target-output of pattern
            NSNumber *_patternTarget = [_pattern.targets objectAtIndex:_outputIndex];
            _outputNet.targetValue   = [_patternTarget doubleValue];
            // MSE
            _errorValue             += _outputNet.outputError * _outputNet.outputError;
            if( _eachOutput )
            {
                _eachOutput(_outputNet);
            }
        }
        //NSLog(@"\n\n");
    }
    
    // RMSE
    _errorValue = sqrt(_errorValue / ( [_patterns count] * [_nets count] ));
    if( _completion )
    {
        _completion(_errorValue);
    }
}

-(void)predicateWithPatterns:(NSArray<KRRBFPattern *> *)_patterns centers:(NSArray<KRRBFCenterNet *> *)_centers outputs:(KRRBFOutputLayerPredication)_outputsBlock
{
    NSMutableDictionary *_predications = [NSMutableDictionary new];
    double _sigma = [self _calculateSigmaWithCenters:_centers];
    for ( KRRBFPattern *_pattern in _patterns )
    {
        NSMutableArray *_rbfValues = [NSMutableArray new];
        for( KRRBFCenterNet *_center in _centers )
        {
            double _rbfValue = [self.activeFunction rbf:_pattern.features x2:_center.features sigma:_sigma];
            [_rbfValues addObject:[NSNumber numberWithDouble:_rbfValue]];
        }
        
        // Predicated outputs of each pattern.
        NSMutableArray *_outputs = [NSMutableArray new];
        for( KRRBFOutputNet *_outputNet in _nets )
        {
            [_outputs addObject:[NSNumber numberWithDouble:[_outputNet outputWithRBFValues:_rbfValues]]];
        }
        
        // Uses pattern.indexKey to record its outputs of predication.
        [_predications setObject:_outputs forKey:_pattern.indexKey];
    }
    
    if( _outputsBlock )
    {
        _outputsBlock(_predications);
    }
}


@end
