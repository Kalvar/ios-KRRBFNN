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
#import "KRRBFFetcher.h"
#import "KRMathLib.h"

/*
 # 有幾個實作想法 :
     1. 把 2 個權重修正方法分開寫 :
     - a). 寫一支 class 用「最小平方法 (LMS)」求權重
     - b). 再寫一支 class 用 SGA 來修正權重
     2. 有幾種用法 :
     - a). OLS 選中心點 -> 用 LMS 解聯立直接求出權重結果 -> 再用 SGA 來做更深層的後續修正，以提昇精度
     - b). OLS 選中心點 -> 亂數給權重 (-0.25 ~ 0.25) -> 再用 SGA 來做修正
     - c). Random 選中心點 -> 用 LMS 解聯立直接求出權重結果 -> 再用 SGA 來做後續修正提昇精度
     - d). Random 選中心點 -> 亂數給權重 (-0.25 ~ 0.25) -> 再用 SGA 來做修正
 
     05/04/2016 已決定以 a ~ d 的方法來實作。
 
 # RBFNN 使用方法 :
     - Recall weights (實作儲存訓練好的 weights，和回復訓練好的 weights)
     - Recall centers (實作儲存訓練好/挑好的 centers，和回復訓練好的 centers)
     - 原來有幾個特徵值與輸出，就要用回幾個特徵值與輸出 (跟一般的 NN 一樣)，否則就要重新訓練網路
 
 # RBFNN that training is 2 steps :
     - 1. To choose initial centers
     - 2. To calculate weights
 */

@interface KRRBFNN ()

@property (nonatomic, strong) KRRBFFetcher *fetcher;

@end

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
        
        _hiddenLayer        = [[KRRBFHiddenLayer alloc] init];
        _outputLayer        = [[KRRBFOutputLayer alloc] init];
        
        _rmse               = 0.0f;
        
        _fetcher            = [[KRRBFFetcher alloc] init];
    }
    return self;
}

#pragma --mark Recover / Other Methods
-(void)recoverForKey:(NSString *)_key
{
    KRRBFPassed *_savedNetwork = [_fetcher objectForKey:_key];
    _hiddenLayer.nets          = _savedNetwork.centers;
    _outputLayer.nets          = _savedNetwork.weights;
}

-(void)removeForKey:(NSString *)_key
{
    [_fetcher removeForKey:_key];
}

-(void)saveForKey:(NSString *)_key
{
    KRRBFPassed *_trainedNetwork = [KRRBFPassed new];
    _trainedNetwork.centers      = self.centers;
    _trainedNetwork.weights      = self.weights;
    [_fetcher save:_trainedNetwork forKey:_key];
}

-(void)reset
{
    [_targets removeAllObjects];
    [_hiddenLayer removeAllCenters];
    [_outputLayer removeAllNets];
    _rmse = 0.0f;
}

-(void)removeCachesIfNeeded
{
    [self.random removeCaches];
}

#pragma --mark Patterns
-(void)addPattern:(KRRBFPattern *)_pattern
{
    [_patterns addObject:_pattern];
}

-(void)addPatterns:(NSArray <KRRBFPattern *> *)_samples
{
    [_patterns addObjectsFromArray:_samples];
}

#pragma --mark Weights
-(void)randomWeightsBetweenMin:(double)_minValue max:(double)_maxValue
{
    if( nil == _patterns || [_patterns count] == 0 )
    {
        return;
    }
    
    if( nil == self.centers || [self.centers count] == 0 )
    {
        return;
    }
    
    KRRBFPattern *_aPattern = [_patterns firstObject];
    // 有幾個期望輸出
    NSInteger _targetCount  = [_aPattern.targets count];
    // 有幾個中心點
    NSInteger _centerCount  = [self.centers count];
    // 權重共有幾條是由 ( 中心點個數 x 期望輸出數量 ) 而來
    [_outputLayer addNetsFromArray:[self.sga randomWeightsWithCenterCount:_centerCount
                                                              targetCount:_targetCount
                                                               betweenMin:_minValue
                                                                      max:_maxValue]];
}

#pragma --mark Training Methods
/*
 * @ Use OLS method to choose initial centers
 *   - (1.0f - _tolerance) is 一般 NN 的收斂誤差
 *   - returns chose centers by OLS
 *
 * @ Parameters
 *   - toSave means to record this chose centers in HiddenLayer as same as self.centers.
 */
-(NSArray <KRRBFCenterNet *> *)pickCentersByOLSWithTolerance:(double)_tolerance toSave:(BOOL)_toSave
{
    // 使用 OLS / LMS 都要在進行 Training 之前先建立 KRRBFTargets
    [self _createTargetsWithPatterns:_patterns];
    self.ols.tolerance = _tolerance;
    // OLS 選取中心點
    NSArray *_choseCenters = [self.ols chooseWithPatterns:_patterns targets:_targets];
    if( _toSave )
    {
        [_hiddenLayer addCentersFromArray:_choseCenters];
    }
    return _choseCenters;
}

-(NSArray <KRRBFCenterNet *> *)pickCentersByOLSWithTolerance:(double)_tolerance
{
    // Default is saving chose centers in HiddenLayer.
    return [self pickCentersByOLSWithTolerance:_tolerance toSave:YES];
}

/*
 * @ Use Random method to choose initial centers
 *   - _limitCount is that how many centers we want to pick ?
 *   - returns chose centers by Random
 *
 * @ Parameters
 *   - toSave means to record this chose centers in HiddenLayer as same as self.centers.
 */
-(NSArray <KRRBFCenterNet *> *)pickCentersByRandomWithLimitCount:(NSInteger)_limitCount toSave:(BOOL)_toSave
{
    NSArray *_choseCenters = [self.random chooseWithPatterns:_patterns pickNumber:_limitCount];
    if( _toSave )
    {
        [_hiddenLayer addCentersFromArray:_choseCenters];
    }
    return _choseCenters;
}

-(NSArray <KRRBFCenterNet *> *)pickCentersByRandomWithLimitCount:(NSInteger)_limitCount
{
    return [self pickCentersByRandomWithLimitCount:_limitCount toSave:YES];
}

-(void)trainLMSWithCompletion:(KRRBFNNCompletion)_completion eachOutput:(KRRBFNNEachOutput)_eachOutput
{
    if( nil == _targets || [_targets count] == 0 )
    {
        [self _createTargetsWithPatterns:_patterns];
    }
    
    // LMS 解聯立一次即求得最佳權重
    [_outputLayer addNetsFromArray:[self.lms outputNetsWithCenters:self.centers patterns:_patterns targets:_targets]];
    [_outputLayer outputWithPatterns:_patterns centers:self.centers completion:^(double rmse) {
        _rmse = rmse;
        if( _completion )
        {
            _completion(YES, self, _rmse);
        }
    } eachOutput:^(KRRBFOutputNet *outputNet) {
        if( _eachOutput )
        {
            _eachOutput(outputNet);
        }
    }];
}

-(void)trainLMSWithCompletion:(KRRBFNNCompletion)_completion
{
    [self trainLMSWithCompletion:_completion eachOutput:nil];
}

-(void)trainSGAWithCompletion:(KRRBFNNCompletion)_completion
{
    // TODO:
    // 跑迭代運算，這裡會不斷的被遞迴直至收斂 (使用 RMSE)
}

#pragma --mark Predication Methods
-(void)predicateWithPatterns:(NSArray <KRRBFPattern *> *)_predicatePatterns output:(KRRBFNNPredication)_outputsBlock
{
    [_outputLayer predicateWithPatterns:_predicatePatterns
                                centers:self.centers
                                outputs:^(NSDictionary<NSString *,NSArray<NSNumber *> *> *predications) {
                                    if( _outputsBlock )
                                    {
                                        _outputsBlock(predications);
                                    }
                                }];
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

#pragma --mark Getters
-(NSMutableArray <KRRBFCenterNet *> *)centers
{
    if( _hiddenLayer )
    {
        return _hiddenLayer.nets;
    }
    return nil;
}

-(NSMutableArray <KRRBFOutputNet *> *)weights
{
    if( _outputLayer )
    {
        return _outputLayer.nets;
    }
    return nil;
}

/*
-(void)dealloc
{
    // Since tested experience, here objects are correctly dealloced.
    NSLog(@"KRRBFNN is dealloced");
}
 */

@end
