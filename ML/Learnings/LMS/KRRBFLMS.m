//
//  KRRBFLms.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/7.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFLMS.h"
#import "KRRBFPattern.h"
#import "KRRBFCenterNet.h"
#import "KRRBFOutputNet.h"
#import "KRRBFTarget.h"
#import "KRRBFActiviation.h"
#import "KRMathLib.h"

@interface KRRBFLMS ()

@property (nonatomic, strong) KRRBFActiviation *activeFunction;
@property (nonatomic, strong) KRMathLib *mathLib;

@end

@implementation KRRBFLMS (LMSSigma)

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

// This method is that normally forward network calculation.
-(NSArray *)_calculatePhiWithPatterns:(NSArray <KRRBFPattern *> *)_patterns toCenters:(NSArray <KRRBFCenterNet *> *)_centers
{
    // Use centers to calculate that sigma.
    double _sigma        = [self _calculateSigmaWithCenters:_centers];
    // That phi[] needs to implement the forward network from patterns to centers.
    NSMutableArray *_phi = [NSMutableArray new];
    for ( KRRBFPattern *_pattern in _patterns )
    {
        NSMutableArray *_rbfDistances = [NSMutableArray new];
        for( KRRBFCenterNet *_center in _centers )
        {
            double _distance = [self.activeFunction rbf:_pattern.features x2:_center.features sigma:_sigma];
            [_rbfDistances addObject:[NSNumber numberWithDouble:_distance]];
        }
        [_phi addObject:_rbfDistances];
    }
    return _phi;
}

// This method is optimized calculation performance without transpose matrix when we are doing solveEquations,
// We could save that transpose matrix processing before we use [_mathLib solveEquationsAtMatrix:outputs:].
-(NSArray *)_calculatePhiWithCenters:(NSArray <KRRBFCenterNet *> *)_centers toPatterns:(NSArray <KRRBFPattern *> *)_patterns
{
    double _sigma        = [self _calculateSigmaWithCenters:_centers];
    NSMutableArray *_phi = [NSMutableArray new];
    for( KRRBFCenterNet *_center in _centers )
    {
        NSMutableArray *_rbfDistances = [NSMutableArray new];
        for ( KRRBFPattern *_pattern in _patterns )
        {
            double _distance = [self.activeFunction rbf:_center.features x2:_pattern.features sigma:_sigma];
            [_rbfDistances addObject:[NSNumber numberWithDouble:_distance]];
        }
        [_phi addObject:_rbfDistances];
    }
    return _phi;
}

@end

@implementation KRRBFLMS

+(instancetype)sharedLMS
{
    static dispatch_once_t pred;
    static KRRBFLMS *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFLMS alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _activeFunction = [KRRBFActiviation sharedActiviation];
        _mathLib        = [KRMathLib sharedLib];
    }
    return self;
}

#pragma --mark Learning Methods
// 使用最小平方法求權重(LMS)
-(NSArray <KRRBFOutputNet *> *)outputNetsWithCenters:(NSArray <KRRBFCenterNet *> *)_centers patterns:(NSArray <KRRBFPattern *> *)_patterns targets:(NSArray <KRRBFTarget *> *)_targets
{
    /*
     * @ Notes :
     *   對 1 個 期望輸出，就要解 1 次聯立，來算出所有的 centers outputs 在到該 target output 時的所有權重線為多少，
     *   所以，有 10 個輸出，就要解 10 次的聯立 (效能超差)。比較好的作法還是用 SGA 來做迭代修正參數，這樣在多輸出的情況下也不用擔心效能太差。

     * @ 效能改進方法 (已實作) :
     *   如果不想讓 MathLib solveEquationsAtMatrix:outputs: 的部份要對 _phi 多做 1 次的轉置矩陣 (取消轉置)，
     *   那就要在使用 _calculatePhiWithPatterns:toCenters: 計算 Phi 的時候，
     *   要從 Patterns to Centers 反過來改成 Centers to Patterns (_calculatePhiWithCenters:toPatterns:)，以中心點為主要對象，
     *   先將同樣中心點所對應到的所有 Patterns 的 phi value，都集中在同一個 Array 裡，這樣就能做到預先轉置矩陣的效果了。
     *
     */
    // 優化算法
    //NSArray *_phi = [self _calculatePhiWithCenters:_centers toPatterns:_patterns;
    
    /*
     * @ 最後決定 :
     *   先保留原公式算法，以便於後人在參照公式的行為上，能保持一致性，之後如有 Performance 需求，再用上述註解的方法優化即可。
     */
    NSArray *_phi            = [self _calculatePhiWithPatterns:_patterns toCenters:_centers];
    NSMutableArray *_weights = [NSMutableArray new];
    NSInteger _outputIndex   = -1;
    for( KRRBFTarget *_targetOutput in _targets )
    {
        _outputIndex              += 1;
        
        // 先解 Output 1, 再解 Output 2 ... 其它類推
        NSArray *_targetWeights    = [_mathLib solveEquationsAtMatrix:_phi outputs:@[_targetOutput.sameSequences]];
        
        KRRBFOutputNet *_outputNet = [[KRRBFOutputNet alloc] init];
        _outputNet.indexKey        = @(_outputIndex); // 第幾個 Output Net
        [_outputNet addWeightsFromArray:_targetWeights];
        
        [_weights addObject:_outputNet];
    }
    return _weights;
}

@end
