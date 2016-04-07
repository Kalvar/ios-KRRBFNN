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

-(NSArray <NSNumber *> *)_calculatePhiWithCenters:(NSArray <KRRBFCenterNet *> *)_centers patterns:(NSArray <KRRBFPattern *> *)_patterns sigma:(double)_sigma
{
    NSMutableArray *_phi = [NSMutableArray new];
    for( KRRBFCenterNet *_center in _centers )
    {
        NSMutableArray *_rbfDistances = [NSMutableArray new];
        for( KRRBFPattern *_pattern in _patterns )
        {
            double _distance = [self.activeFunction rbf:_center.features x2:_pattern.features sigma:_sigma];
            [_rbfDistances addObject:[NSNumber numberWithDouble:_distance]]; // phi(j,i)
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
-(NSArray *)weightsWithCenters:(NSArray <KRRBFCenterNet *> *)_centers patterns:(NSArray <KRRBFPattern *> *)_patterns targets:(NSArray <KRRBFTarget *> *)_targets
{
    double _sigma = [self _calculateSigmaWithCenters:_centers];
    NSArray *_phi = [self _calculatePhiWithCenters:_centers patterns:_patterns sigma:_sigma];
    
    // 對 1 個 期望輸出 就要解 1 次聯立，來算出所有的 centers outputs 在到該 target output 時的所有權重線為多少，
    // 所以，有 10 個輸出，就要解 10 次的聯立 (效能超差)。比較好的作法還是用 SGA 來做迭代修正參數，這樣在多輸出的情況下也不用擔心效能太差。
    // 這裡還是把 LMS 補完，再來寫 SGA -> 最後是 Random Choice
    NSMutableArray *_weights = [NSMutableArray new];
    for( KRRBFTarget *_targetOutput in _targets )
    {
        //Crashed here, it seems _targetOutput.sameSequences that dimesions are not match parameters ... ?
        
        NSArray *_targetWeights = [_mathLib solveEquationsAtMatrix:_phi outputs:_targetOutput.sameSequences];
        [_weights addObject:_targetWeights];
    }
    
    return _weights;
}

@end
