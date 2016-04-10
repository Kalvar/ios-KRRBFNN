//
//  KRRBFNN.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFNN.h"
#import "KRRBFTarget.h"

@implementation KRRBFNN (trainingSteps)

// 製作 KRRBFTarget 並設定 sameSequences 以供 OLS / LMS 計算使用
-(void)_createTargetsWithPatterns:(NSArray <KRRBFPattern *> *)_patterns
{
    NSMutableArray *_targets = self.targets;
    if( nil != _targets && [_targets isKindOfClass:[NSMutableArray class]] )
    {
        [_targets removeAllObjects];
    }
    else
    {
        _targets = [NSMutableArray new];
    }
    // 先 Loop 所有的 Patterns Outputs，之後取出同維度的所有 Target Output 集合在一起 :
    // 先取出有幾個期望輸出 (幾個分類)
    NSInteger _targetCount = [[[_patterns firstObject] targets] count];
    // 再依序 Loop 所有的 Patterns 將它們的期望輸出值依序放入各自同維度的陣列裡集合起來
    for( NSInteger i=0; i<_targetCount; i++ )
    {
        KRRBFTarget *_targetOutput = [[KRRBFTarget alloc] init];
        for( KRRBFPattern *p in _patterns )
        {
            [_targetOutput.sameSequences addObject:[p.targets objectAtIndex:i]];
        }
        [_targets addObject:_targetOutput];
    }
}

@end

@implementation KRRBFNN

+(instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static KRRBFNN *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFNN alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _patterns           = [NSMutableArray new];
        _targets            = [NSMutableArray new];
        _centers            = [NSMutableArray new];
        _weights            = [NSMutableArray new];
        
        //_centerChoiceMethod = KRRBFNNCenterChoiceOLS;
        //_learningMethod     = KRRBFNNLearningLMS;
    }
    return self;
}

-(void)addPattern:(KRRBFPattern *)_pattern
{
    [_patterns addObject:_pattern];
}

-(void)addPatterns:(NSArray <KRRBFPattern *> *)_samples
{
    [_patterns addObjectsFromArray:_samples];
}

#pragma --mark Training Methods
// RBFNN that training is 2 steps :
// 1. To choose initial centers
// 2. To calculate weights

/*
 * @ Use OLS method to choose initial centers
 *   - (1.0f - _tolerance) is 一般 NN 的收斂誤差
 *   - returns chose centers by OLS
 */
-(NSArray <KRRBFCenterNet *> *)pickingCentersByOLSWithTolerance:(double)_tolerance setToCenters:(BOOL)_setToCenters
{
    [self _createTargetsWithPatterns:_patterns];
    self.ols.tolerance = _tolerance;
    // OLS 選取中心點
    NSArray *_choseCenters = [self.ols chooseWithPatterns:_patterns targets:_targets];
    if( _setToCenters )
    {
        [_centers removeAllObjects];
        [_centers addObjectsFromArray:_choseCenters];
    }
    return _choseCenters;
}

/*
 * @ Use Random method to choose initial centers
 *   - _limitCount is that how many centers we want to pick ?
 *   - returns chose centers by Random
 */
-(NSArray <KRRBFCenterNet *> *)pickingCentersByRandomWithLimitCount:(NSInteger)_limitCount setToCenters:(BOOL)_setToCenters
{
    // TODO:
    return nil;
}

-(void)trainingByLMSWithCompletion:(KRRBFNNCompletion)_completion
{
    if( nil == _targets || [_targets count] == 0 )
    {
        [self _createTargetsWithPatterns:_patterns];
    }
    // LMS 解聯立一次即求得最佳權重
    NSArray *_newWeights = [self.lms weightsWithCenters:_centers patterns:_patterns targets:_targets];
    [_weights removeAllObjects];
    [_weights addObjectsFromArray:_newWeights];
    if( _completion )
    {
        _completion(YES, self);
    }
}

-(void)trainingBySGAWithCompletion:(KRRBFNNCompletion)_completion
{
    // TODO:
    // 跑迭代運算，這裡會不斷的被遞迴直至收斂 (使用 RMSE)
}

#pragma --mark Getters
-(KRRBFOLS *)ols
{
    if( nil == _ols )
    {
        _ols = [[KRRBFOLS alloc] init];
    }
    return _ols;
}

-(KRRBFRandom *)random
{
    if( nil == _random )
    {
        _random = [[KRRBFRandom alloc] init];
    }
    return _random;
}

-(KRRBFLMS *)lms
{
    if( nil == _lms )
    {
        _lms = [[KRRBFLMS alloc] init];
    }
    return _lms;
}

-(KRRBFSGA *)sga
{
    if( nil == _sga )
    {
        _sga = [[KRRBFSGA alloc] init];
    }
    return _sga;
}



@end
