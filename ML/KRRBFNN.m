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
#import "KRRBFHiddenLayer.h"
#import "KRRBFOutputLayer.h"
#import "KRMathLib.h"

@interface KRRBFNN ()

// Center choice methods
@property (nonatomic, strong) KRRBFOLS *ols;
@property (nonatomic, strong) KRRBFRandom *random;

// Learning methods
@property (nonatomic, strong) KRRBFLMS *lms;
@property (nonatomic, strong) KRRBFSGA *sga;

@property (nonatomic, strong) KRRBFFetcher *fetcher;

@property (nonatomic, strong) KRRBFHiddenLayer *hiddenLayer;
@property (nonatomic, strong) KRRBFOutputLayer *outputLayer;

@end

@implementation KRRBFNN (trainingOLS)

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

@implementation KRRBFNN (trainingSGA)

// 迭代的控制、收斂與否，都是在這裡 RBFNN 裡控制的，不是在 SGA 裡做，因為 SGA 本身只是「修正方法」，不做其它邏輯的控制，
// 要跑到這裡的 training code 的話，就必須先在外部把 weights 設定好，
// 例如，已經先使用 LMS 解過權重了、Recover 儲存的 Network Weights、使用 Random 設定 Weights 等。
-(void)_sgaWithCompletion:(KRRBFNNCompletion)_completion iteration:(KRRBFNNIteration)_iteration
{
    if( nil == self.weights )
    {
        return;
    }
    
    self.iterationTimes += 1;
    __weak typeof(self) _weakSelf = self;
    [self.outputLayer outputWithPatterns:self.patterns centers:self.centers completion:^(KRRBFOutputLayer *layer) {
        
        __strong typeof(self) _strongSelf = _weakSelf;
        // 如果已達收斂條件，就不再進行迭代運算 (returns NO 會接著解發 completion block)
        if( _strongSelf.iterationTimes >= _strongSelf.maxIteration || layer.rmse <= _strongSelf.toleranceError )
        {
            if( _completion )
            {
                _completion(YES, self);
            }
            [_strongSelf.sga freeReferences];
            return;
        }
        else if( _iteration )
        {
            BOOL _isContinue = _iteration(_strongSelf.iterationTimes, layer.rmse);
            if( !_isContinue )
            {
                if( _completion )
                {
                    _completion(NO, self);
                }
                [_strongSelf.sga freeReferences];
                return;
            }
        }
        
        // 繼續遞迴跑下一迭代
        [_strongSelf _sgaWithCompletion:_completion iteration:_iteration];
        
    } patternOutput:^(NSArray<KRRBFOutputNet *> *patternOutputs, KRRBFPattern *currentPattern) {
        __strong typeof(self) _strongSelf = _weakSelf;
        // Below updating methods are used reference memory to automatic update outside values.
        [_strongSelf.sga updateCentersWithPattern:currentPattern];
        [_strongSelf.sga updateWeights];
    }];
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
        
        _maxIteration       = 1;
        _toleranceError     = 0.001f;
        _learningRate       = 1.0f;
        
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

#pragma --mark Weights & Centers
// Random to setup weights of network must after picked centers and added patterns.
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

-(void)addCenters:(NSArray <KRRBFCenterNet *> *)_objects
{
    if( _hiddenLayer )
    {
        [_hiddenLayer addCentersFromArray:_objects];
    }
}

-(void)addWeights:(NSArray <KRRBFOutputNet *> *)_objects
{
    if( _outputLayer )
    {
        [_outputLayer addNetsFromArray:_objects];
    }
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
        [self addCenters:_choseCenters];
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
        [self addCenters:_choseCenters];
    }
    return _choseCenters;
}

-(NSArray <KRRBFCenterNet *> *)pickCentersByRandomWithLimitCount:(NSInteger)_limitCount
{
    return [self pickCentersByRandomWithLimitCount:_limitCount toSave:YES];
}

-(void)trainLMSWithCompletion:(KRRBFNNCompletion)_completion patternOutput:(KRRBFNNPatternOutput)_patternOutput
{
    if( nil == _targets || [_targets count] == 0 )
    {
        [self _createTargetsWithPatterns:_patterns];
    }
    
    // LMS 解聯立一次即求得最佳權重
    [_outputLayer addNetsFromArray:[self.lms outputNetsWithCenters:self.centers patterns:_patterns targets:_targets]];
    [_outputLayer setupCommonSigmaWithCenters:self.centers];
    [_outputLayer outputWithPatterns:_patterns centers:self.centers completion:^(KRRBFOutputLayer *layer) {
        if( _completion )
        {
            _completion(YES, self);
        }
    } patternOutput:^(NSArray<KRRBFOutputNet *> *patternOutputs, KRRBFPattern *currentPattern) {
        if( _patternOutput )
        {
            _patternOutput(patternOutputs);
        }
    }];
}

-(void)trainLMSWithCompletion:(KRRBFNNCompletion)_completion
{
    [self trainLMSWithCompletion:_completion patternOutput:nil];
}

-(void)trainSGAWithCompletion:(KRRBFNNCompletion)_completion iteration:(KRRBFNNIteration)_iteration
{
    _iterationTimes  = 0;
    
    // Setups that reference objects with SGA.
    KRRBFSGA *sga    = self.sga;
    sga.learningRate = self.learningRate;
    sga.patterns     = self.patterns;
    sga.centers      = self.centers;
    sga.weights      = self.weights;
    // Setups initial common sigma of all centers (important step, it must be have).
    [_outputLayer setupCommonSigmaWithCenters:sga.centers];
    [self _sgaWithCompletion:_completion iteration:_iteration];
}

-(void)trainSGAWithCompletion:(KRRBFNNCompletion)_completion
{
    [self trainSGAWithCompletion:_completion iteration:nil];
}

#pragma --mark Predication Methods
-(void)predicateWithPatterns:(NSArray <KRRBFPattern *> *)_predicatePatterns output:(KRRBFNNPredication)_outputsBlock
{
    [_outputLayer predicateWithPatterns:_predicatePatterns
                                centers:self.centers
                                outputs:^(NSDictionary<NSString *, NSArray<NSNumber *> *> *predications) {
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

-(double)rmse
{
    if( _outputLayer )
    {
        return _outputLayer.rmse;
    }
    return 0.0f;
}

//*
-(void)dealloc
{
    // Since tested experience, here objects are correctly dealloced.
    NSLog(@"KRRBFNN is dealloced");
}
 //*/

@end
