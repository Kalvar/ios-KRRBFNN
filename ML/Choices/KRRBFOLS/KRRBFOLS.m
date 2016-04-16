//
//  KRRBFOLS.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFOLS.h"
#import "KRMathLib.h"
#import "KRRBFActiviation.h"
#import "KRRBFPattern.h"
#import "KRRBFTarget.h"
#import "KRRBFCenterNet.h"

@interface KRRBFOLS ()

@property (nonatomic, strong) KRRBFActiviation *activeFunction;
@property (nonatomic, strong) KRMathLib *mathLib;

@end

@implementation KRRBFOLS (candidatesDistance)
// 計算 Patterns 和 Centers 之間的距離並取出最大的距離值
// Centers are training samples or chose centers of used OLS.
-(double)_calculateMaxDistanceWithCenters:(NSArray <KRRBFPattern *> *)_centers
{
    double _maxDistance = -1.0f;
    // Looping all patterns (we used it to be centers)
    for( KRRBFPattern *_center in _centers )
    {
        // One pattern to all centers
        for( KRRBFPattern *_candidate in _centers )
        {
            double _distance = [self.activeFunction euclidean:_center.features x2:_candidate.features];
            // 取出最大距離 (用於後續計算 Sigma)
            if( _distance > _maxDistance )
            {
                _maxDistance = _distance;
            }
        }
    }
    return _maxDistance;
}

-(double)_calculateSigmaWithCenters:(NSArray <KRRBFPattern *> *)_centers
{
    double _maxDistance = [self _calculateMaxDistanceWithCenters:_centers];
    return (_maxDistance >= 0.0f) ? _maxDistance / sqrt([_centers count]) : 0.0f;
}

//#define NumberArray NSArray<NSNumber *>
-(NSMutableArray *)_calculatePhiWithCenters:(NSArray <KRRBFPattern *> *)_centers patterns:(NSArray <KRRBFPattern *> *)_patterns sigma:(double)_sigma
{
    // 用算好的 Sigma 和活化函式來重新運算每一筆 Pattern 和 Centers 之間的距離 (RBF, phi)
    NSMutableArray *_phi = [NSMutableArray new];
    for( KRRBFPattern *_center in _centers )
    {
        // 先將每一個 Pattern 對同一個 Center 的 RBF Distance 記起來
        NSMutableArray *_rbfDistances = [NSMutableArray new];
        for( KRRBFPattern *_input in _patterns )
        {
            double _distance = [self.activeFunction rbf:_center.features x2:_input.features sigma:_sigma];
            [_rbfDistances addObject:[NSNumber numberWithDouble:_distance]]; // phi(j,i)
        }
        // 記錄該 Center 對所有 Patterns 的 RBF 距離, 例如 Center1 對 10 個 Patterns (這裡在建構 5.17 的 R 矩陣)
        [_phi addObject:_rbfDistances];
    }
    return _phi;
}

@end

@implementation KRRBFOLS

+(instancetype)sharedOLS
{
    static dispatch_once_t pred;
    static KRRBFOLS *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFOLS alloc] init];
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
        _maxPick        = 0; // Default
    }
    return self;
}

// _targets 是目標輸出值，書本公式裡的 d，Sample Code 裡的 t
-(NSArray <KRRBFCenterNet *> *)chooseWithPatterns:(NSArray<KRRBFPattern *> *)_patterns targets:(NSArray<KRRBFTarget *> *)_targets
{
    NSMutableArray <KRRBFCenterNet *> *_choseCenters = [NSMutableArray new];
    
    // Copy a pattern array to do continue calculation.
    NSMutableArray *_trainSamples = [_patterns mutableCopy];
    
    // 運算原理是每一個 Pattern 都是 Center 的 Candidate，也就是每一個 Pattern 都有可能成為 Center，
    // 這裡開始對每一個中心點做交互運算，例如有 300 筆 Training Patterns，這裡就要跑 300 * 300 = 9 萬次去計算所有 Patterns 彼此之間的距離，
    // 先取出 Patterns 和 Candidate Centers 之間運算完後最大的距離值再計算 Sigma (標準差)。
    // Sigma = 所有中心點間最大的距離 / sqrt(範本數量)，另 1 個公式是 / sqrt(範本數量 + 1)
    double _sigma = [self _calculateSigmaWithCenters:_trainSamples];
    // 用算好的 Sigma 和活化函式來重新運算每一筆 Pattern 和 Centers 之間的距離 (RBF)
    // _centerRBFDistances is phi.
    NSMutableArray *_centerRBFDistances = [self _calculatePhiWithCenters:_trainSamples patterns:_trainSamples sigma:_sigma];
    
    /*
     * @ Theory
     *   - 多分類( Multi-Class )作法 :
     *     - 照原公式，先 1 顆 1 顆 Output Net 的算 err ( 誤差下降率縮寫, Error Reduction Rate )，之後累加該 Center(i) 的 err value 後，
     *       取其平均值，再依序選出 err 最大的值做 New Center。
     *     - 如果不取平均，而是直接用 err 的累加值去做判斷，這會讓 tolerance ( 收斂誤差 ; 正確率 ) 也要跟著擴大和調整 
     *       ( ex : 10 顆 Output Nets 合計的 err 是很大的 )，否則很快就會達到 tolerance 的值而收斂。
     */
    // 求每一個 Center 的誤差下降率 (ERR, Error Reduction Rate)(5.25 和下面的函式) 與最大下降誤差值 (下降越大代表誤差減少越多)
    NSMutableArray *_errors   = [NSMutableArray new];
    NSInteger _maxErrIndex    = -1;
    double _maxErrValue       = 0.0f;
    // 先依序列舉出每一個 Center 以求得 RBFNN 的第 1 個初始中心點
    NSInteger _centerNumber   = -1;
    // 有幾個分類的期望輸出
    NSInteger _countTarget    = [_targets count];
    for( NSArray *_centerRBFs in _centerRBFDistances )
    {
        _centerNumber += 1;
        // @ The formula is following these below steps :
        // _s is RBF values of specified center
        NSArray *_s    = _centerRBFs;
        double _h      = [_mathLib sumMatrix:_s anotherMatrix:_s];
        // 針對多分類做算法修改 (原公式只有單分類輸出) :
        // 先算出該中心點對所有訓練範本其相同維度的期望輸出值的所有誤差下降率總和，
        // 再取其平均值，以求得該中心點對所有期望輸出的平均誤差下降率( Error Reduction Rate )。
        double _averageErr = 0.0f;
        for( KRRBFTarget *_olsTarget in _targets )
        {
            // 計算目標輸出值的內積值(Inner Product) t*t'
            NSArray *_sampleTargets = _olsTarget.sameSequences;
            // _t is whole targets inner product
            double _t               = [_mathLib sumMatrix:_sampleTargets anotherMatrix:_sampleTargets];
            double _g               = [_mathLib sumMatrix:_s anotherMatrix:_sampleTargets] / _h;
            double _errValue        = _h * (_g * _g) / _t;
            [_errors addObject:[NSNumber numberWithDouble:_errValue]];
            _averageErr            += _errValue;
        }
        // To calculate that average err value of specified center.
        _averageErr /= _countTarget;
        // Then compare to get MAX(err of center)
        if( _maxErrIndex < 0 || _averageErr > _maxErrValue )
        {
            _maxErrIndex = _centerNumber;
            _maxErrValue = _averageErr;
        }
    }
    
    NSInteger _k                 = 0;
    // 取出第 1 個挑選到的新中心點
    KRRBFPattern *_pickedPattern = [_trainSamples objectAtIndex:_maxErrIndex];
    // Copy that chose pattern to be a new center.
    KRRBFCenterNet *_newCenter   = [[KRRBFCenterNet alloc] initWithFeatures:_pickedPattern.features];
    _newCenter.indexKey          = @(_k);
    [_choseCenters addObject:_newCenter];
    
    // 再想想這裡的流程要怎麼重新設計，參考書本 P.185
    NSMutableArray *_ss      = [NSMutableArray new];
    [_ss addObject:[_centerRBFDistances objectAtIndex:_maxErrIndex]];
    double _sumError         = _maxErrValue;
    [_trainSamples removeObjectAtIndex:_maxErrIndex];
    [_centerRBFDistances removeObjectAtIndex:_maxErrIndex];
    [_errors removeAllObjects];
    
    // 再來開始挑選其它的中心點 (如果第 1 個中心點未滿足收斂需求的話)
    // Alphaik = (Si x Uk) / (Si x Si)
    // Sk = Uk - Sum(Alphaik x Si), for Sum scope is i=1 ... to k-1
    NSInteger _patternCount = [_trainSamples count];
    // 如果正確率還不到收斂範圍 && 還沒有 Loop 完所有剩餘的 Patterns
    while( _sumError<_tolerance && _k<_patternCount )
    {
        // 或是取到限定的 Centers 數量就停止，_k+1 是因為 _k 從零開始且一開始就有先選擇了 1 個中心點
        if( _maxPick > 0 && _k+1 >= _maxPick )
        {
            break;
        }
        
        NSInteger _maxErrIndex = -1;
        double _maxErrValue    = 0.0f;
        
        _k += 1;
        // _i is Uk that another RBF Center picker index
        // 列舉每一個還沒被選上當中心點的 Pattern 來做運算
        for( NSInteger _i=0; _i<_patternCount-_k+1; _i++ )
        {
            // The _s is current uk (same as phi), here must use "copy" to avoid memory reference
            NSArray *_s = [[_centerRBFDistances objectAtIndex:_i] copy];
            // 開始計算並讓 _phi 連減 alpha(k) 與 S(i) 互乘的值
            // _j is chose Center picker index
            for( NSInteger _j=0; _j<_k-1; _j++ )
            {
                // 這裡的 Code 以 P.179 的 5.16 公式為主 (參考該頁上方的 S2, S3 範例)，而後 P.185 的 phi(:, i) - SS * a(:, i) 已解出，就照 5.16 公式一樣，
                // SS 是 m x n 矩陣，a(:, i) 為 m x 1 矩陣，運算方式是 phi(:, i) - ( SS[0] x a(:, i)[0] + SS[0] x a(:, i)[1] ) ... (用連減也可以)，
                
                // formula : alphaValue = (SS(:, j)' * phi(:, i)) / (SS(:, j)' * SS(:, j))
                NSArray *_ssj      = [_ss objectAtIndex:_j];
                NSArray *_phi      = [_centerRBFDistances objectAtIndex:_i];
                
                // (取出 RBF 中心點 * 另一個 RBF 中心點) / (RBF 中心點 * RBF 中心點)
                double _alphaValue = [_mathLib sumMatrix:_ssj anotherMatrix:_phi] / [_mathLib sumMatrix:_ssj anotherMatrix:_ssj];
                
                // 在這裡將 sj * alpha (即公式裡的 si)
                NSArray *_alphaSi  = [_mathLib multiplyMatrix:_ssj byNumber:_alphaValue];
                
                // 連減 SUM(alpha(ik) * s(i))
                _s = [_mathLib minusMatrix:_s anotherMatrix:_alphaSi];
            }
            
            double _h          = [_mathLib sumMatrix:_s anotherMatrix:_s];
            double _averageErr = 0.0f;
            for( KRRBFTarget *_olsTarget in _targets )
            {
                NSArray *_sampleTargets = _olsTarget.sameSequences;
                double _t               = [_mathLib sumMatrix:_sampleTargets anotherMatrix:_sampleTargets];
                double _g               = [_mathLib sumMatrix:_s anotherMatrix:_sampleTargets] / _h;
                double _errValue        = _h * (_g * _g) / _t;
                [_errors addObject:[NSNumber numberWithDouble:_errValue]];
                _averageErr            += _errValue;
            }
            _averageErr /= _countTarget;
            if( _maxErrIndex < 0 || _averageErr > _maxErrValue )
            {
                _maxErrIndex = _i;
                _maxErrValue = _averageErr;
            }
        }
        
        //NSLog(@"sumErr %f, _maxErrValue %f", _sumError, _maxErrValue);
        
        // 取出這次誤差下降率最大的中心點
        KRRBFPattern *_pickedPattern = [_trainSamples objectAtIndex:_maxErrIndex];
        // Copy that chose pattern to be a new center.
        KRRBFCenterNet *_newCenter   = [[KRRBFCenterNet alloc] initWithFeatures:_pickedPattern.features];
        _newCenter.indexKey          = @(_k);
        [_choseCenters addObject:_newCenter];
        
        // For next picking usage.
        [_ss addObject:[_centerRBFDistances objectAtIndex:_maxErrIndex]];
        _sumError               += _maxErrValue;
        [_trainSamples removeObjectAtIndex:_maxErrIndex];
        [_centerRBFDistances removeObjectAtIndex:_maxErrIndex];
        [_errors removeAllObjects];
        
    } // end while
    
    //NSLog(@"選中的 %li 個 Centers : %@", [_choseCenters count], _choseCenters);
    
    return _choseCenters;
}

/*
-(void)dealloc
{
    NSLog(@"KRRBFOLS is dealloced");
}
 */

@end
