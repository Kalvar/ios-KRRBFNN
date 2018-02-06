//
//  KRRBFSga.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/7.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFSGA.h"
#import "KRRBFPattern.h"
#import "KRRBFCenterNet.h"
#import "KRRBFOutputNet.h"
#import "KRMathLib.h"

@interface KRRBFSGA ()

@property (nonatomic, strong) KRMathLib *mathLib;

@end

@implementation KRRBFSGA

+(instancetype)sharedSGA
{
    static dispatch_once_t pred;
    static KRRBFSGA *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFSGA alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _mathLib            = [KRMathLib sharedLib];
        _weightLearningRate = 1.0f;
        _centerLearningRate = 1.0f;
        _sigmaLearningRate  = 1.0f;
        
        // 這裡刻意使用 memory reference 來讓 centers, weights, patterns 都會在更新時，一併連動外部的參數一起修改，如此能更有效率的運算，
        // 如果這樣的記憶體連動方法不妥，以後再於重構時進行優化。
        _centers            = nil;
        _weights            = nil;
        _patterns           = nil;
    }
    return self;
}

#pragma --mark Weights & Output Nets
// 隨機設定權重
-(NSArray <KRRBFOutputNet *> *)randomWeightsWithCenterCount:(NSInteger)_centerCount targetCount:(NSInteger)_targetCount betweenMin:(double)_minValue max:(double)_maxValue
{
    NSMutableArray *_randomWeights = [NSMutableArray new];
    for( NSInteger _i=0; _i<_targetCount; _i++ )
    {
        KRRBFOutputNet *_outputNet = [[KRRBFOutputNet alloc] init];
        _outputNet.indexKey        = @(_i);
        for( NSInteger _j=0; _j<_centerCount; _j++ )
        {
            // If maxValue && minValue are 0.0f that weight also be zero.
            [_outputNet addWeight:[NSNumber numberWithDouble:[_mathLib randomDoubleMax:_maxValue min:_minValue]]];
        }
        [_randomWeights addObject:_outputNet];
    }
    return _randomWeights;
}

// 設定全零的權重
-(NSArray <KRRBFOutputNet *> *)zeroWeightsWithCenterCount:(NSInteger)_centerCount targetCount:(NSInteger)_targetCount
{
    return [self randomWeightsWithCenterCount:_centerCount targetCount:_targetCount betweenMin:0.0f max:0.0f];
}

#pragma --mark Update Methods
// 先更新的中心點與 Sigma
-(void)updateCentersWithPattern:(KRRBFPattern *)_currentPattern
{
    NSInteger _index = -1;
    for( KRRBFCenterNet *_centerNet in _centers )
    {
        _index += 1;
        // 運算原則是"更新中心點的同時，一併更新該中心點的 Sigma"
        // 計算 Output Error ( ei(p) = di(p) - yi(p) ), 即計算該 Center 要負擔多少的 Output Error Value
        // SUM( wj(p) * ei(p) ), ex : c1 error = ( error1 * weight11 + error2 * weight12 + ... + errorN * weights1N ) / ( sigma * sigma )
        // -> _sumCost for updating centers and their sigmas.
        double _sumCost = 0.0f;
        for( KRRBFOutputNet *_outputNet in _weights )
        {
            NSNumber *_centerWeight = [_outputNet.weights objectAtIndex:_index];
            _sumCost += [_centerWeight doubleValue] * _outputNet.outputError; // Using the output error without ^2, it is ei(p) not the E(p).
        }
        double _deltaCenterValue = _centerLearningRate * ( _sumCost / pow(_centerNet.sigma, 2) ) * _centerNet.rbfValue; // learningRate * ( wj(p) * ei(p) / sigma^2(p) ) * R(||x(p) - cj(p)||)
        NSArray *_minusFeatures  = [_mathLib minusMatrix:_currentPattern.features anotherMatrix:_centerNet.features]; // ( x(p) - cj(p) )
        NSArray *_deltaCenters   = [_mathLib multiplyMatrix:_minusFeatures byNumber:_deltaCenterValue];
        NSArray *_newCenters     = [_mathLib plusMatrix:_centerNet.features anotherMatrix:_deltaCenters];
        
        // sigma error = ( error1 * weight11 + error2 * weight12 + ... + errorN * weights1N ) / ( sigma * sigma * sigma )
        // distance    = ||x(p) - cj(p)||^2 that means euclidean without sqrt(), the cj(p) is current old features of center (not updated).
        double _distance   = [_mathLib distance:_currentPattern.features x2:_centerNet.features];
        double _deltaSigma = _sigmaLearningRate * ( _sumCost / pow(_centerNet.sigma, 3) ) * _centerNet.rbfValue * _distance;
        double _newSigma   = _centerNet.sigma + _deltaSigma;
        _centerNet.sigma   = _newSigma;
        
        // Updating Centers
        [_centerNet addFeaturesFromArray:_newCenters];
    }
}

// 最後再更新權重 (因為舊權重會先不斷的被共用於中心點和 Sigma 的更新運算上，故最後才能更新權重)
-(void)updateWeights
{
    // 用 OutputNet (輸出神經元) 的輸出誤差值(不是平方值 E(p))來修正跟這一個 OutputNet 相連接的每一條權重線
    for( KRRBFOutputNet *_outputNet in _weights )
    {
        NSMutableArray *_newWeights = [NSMutableArray new];
        double _errorValue          = _outputNet.outputError;
        // 取出所有對應該 Output Net 的中心點
        NSInteger _index = -1;
        for( KRRBFCenterNet *_centerNet in _centers )
        {
            _index += 1;
            // _centerNet.rbfValue is 對應當前 Output Net 的 "各個中心點的 RBF output value"
            // 取出 Output Net 對應該 Center 的權重
            NSNumber *_centerWeight = [_outputNet.weights objectAtIndex:_index];
            double _newWeight       = [_centerWeight doubleValue] + (_weightLearningRate * _errorValue * _centerNet.rbfValue);
            [_newWeights addObject:[NSNumber numberWithDouble:_newWeight]];
        }
        [_outputNet addWeightsFromArray:_newWeights];
    }
}

#pragma --mark Free Memory
-(void)freeReferences
{
    // Could set nil to release the weak reference then help ARC to early recycle that memory.
    _centers  = nil;
    _weights  = nil;
    _patterns = nil;
}

#pragma --mark Setters
-(void)setLearningRate:(double)_rate
{
    _weightLearningRate = _rate;
    _centerLearningRate = _rate;
    _sigmaLearningRate  = _rate;
    _learningRate       = _rate;
}

-(void)dealloc
{
    //NSLog(@"KRRBFSGA is dealloced");
}

@end
