//
//  KRRBFNN.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFNN.h"
#import "KRRBFTarget.h"
#import "KRRBFOutputLayer.h"

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
        
        _hiddenLayer        = [KRRBFHiddenLayer sharedLayer];
        _outputLayer        = [KRRBFOutputLayer sharedLayer];
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
        [_hiddenLayer addCentersFromArray:_choseCenters];
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
    NSArray *_outputWeights = [self.lms outputWeightsWithCenters:self.centers patterns:_patterns targets:_targets];
    [_outputLayer addNetsFromArray:_outputWeights];
    [_outputLayer outputWithPatterns:_patterns centers:self.centers eachOutput:^(KRRBFOutputNet *outputNet) {
        NSLog(@"net(%@) the output is %f and target is %f", outputNet.indexKey, outputNet.outputValue, outputNet.targetValue);
    } completion:^(double rmse) {
        NSLog(@"rmse : %f", rmse);
        if( _completion )
        {
            _completion(YES, self);
        }
    }];
    
    //NSLog(@"_weights : %@", self.weights);
}

-(void)trainingBySGAWithCompletion:(KRRBFNNCompletion)_completion
{
    // TODO:
    // 跑迭代運算，這裡會不斷的被遞迴直至收斂 (使用 RMSE)
}

#pragma --mark Getters Learning Methods
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

#pragma --mark Getters Parameters
-(NSArray <KRRBFCenterNet *> *)centers
{
    if( _hiddenLayer )
    {
        return _hiddenLayer.nets;
    }
    return nil;
}

-(NSArray <KRRBFOutputNet *> *)weights
{
    if( _outputLayer )
    {
        return _outputLayer.nets;
    }
    return nil;
}



// 使用 OLS / LMS 都要在進行 Training 之前先建立 KRRBFTargets
-(void)createTargets
{
    
    /*
# 有幾個實作想法 :
    1. 把 2 個權重修正方法分開寫 :
    - a). 寫一支 class 用「最小平方法 (LMS)」求權重
    - b). 再寫一支 class 用 SGA 來修正權重
    2. 有幾種用法 :
    - a). OLS 選中心點 -> 用 LMS 解聯立直接求出權重結果 -> 再用 SGA 來做後續修正提昇精度
    - b). OLS 選中心點 -> 亂數給權重 (-0.25 ~ 0.25) -> 再用 SGA 來做修正
    - c). Random 選中心點 -> 用 LMS 解聯立直接求出權重結果 -> 再用 SGA 來做後續修正提昇精度
    - d). Random 選中心點 -> 亂數給權重 (-0.25 ~ 0.25) -> 再用 SGA 來做修正
    
    05/04/2016 已決定以 a ~ d 的方法來實作。
    
# RBFNN 使用方法 :
    - Recall weights (實作儲存訓練好的 weights，和回復訓練好的 weights)
    - Recall centers (實作儲存訓練好/挑好的 centers，和回復訓練好的 centerss)
    - 原來有幾個輸出，就要用回幾個輸出 (跟一般的 NN 一樣)
    
    */

}

@end
