//
//  KRRBFOLS.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFOLS.h"
#import "KRRBFActiviation.h"
#import "KRRBFPattern.h"
#import "KRRBFCandidateCenter.h"
#import "KRMathLib.h"

static NSString *kKRRBFOLSCalculatedDistancesKey   = @"distances";
static NSString *kKRRBFOLSCalculatedMaxDistanceKey = @"maxDistance";

@interface KRRBFOLS ()

@property (nonatomic, strong) KRRBFActiviation *activeFunction;
@property (nonatomic, strong) KRMathLib *mathLib;

@end

@implementation KRRBFOLS (calculateDistance)

// 除了計算 Patterns 和 Centers 之間的距離外，同步取出最大的距離值
-(NSDictionary *)_calculateCenterDistancesWithPatterns:(NSArray *)_patterns
{
    double _maxDistance               = -1.0f;
    NSMutableDictionary *_withCenters = [NSMutableDictionary new];
    for( KRRBFPattern *_pattern in _patterns )
    {
        NSMutableArray *_centerDistances = [NSMutableArray new];
        for( KRRBFPattern *_candidate in _patterns )
        {
            double _distance                       = [self.activeFunction euclidean:_pattern.features x2:_candidate.features];
            KRRBFCandidateCenter *_candidateCenter = [[KRRBFCandidateCenter alloc] init];
            [_candidateCenter recordPatternIndex:_pattern.number centerIndex:_candidate.number distance:_distance];
            [_centerDistances addObject:_candidateCenter];
            if( _distance > _maxDistance )
            {
                _maxDistance = _distance;
            }
        }
        [_withCenters setObject:_centerDistances forKey:_pattern.number];
    }
    return @{kKRRBFOLSCalculatedDistancesKey   : _withCenters,
             kKRRBFOLSCalculatedMaxDistanceKey : [NSNumber numberWithDouble:_maxDistance]};
}

-(NSDictionary *)_fetchCenterDistancesByResults:(NSDictionary *)_results
{
    return [_results objectForKey:kKRRBFOLSCalculatedDistancesKey];
}

-(NSNumber *)_fetchMaxDistanceByResults:(NSDictionary *)_results
{
    return [_results objectForKey:kKRRBFOLSCalculatedMaxDistanceKey];
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
        _centers        = [NSMutableArray new];
        _activeFunction = [KRRBFActiviation sharedActiviation];
        _mathLib        = [KRMathLib sharedLib];
    }
    return self;
}

// _targets 是目標輸出值，書本公式裡的 d，Sample Code 裡的 t
-(void)olsWithPatterns:(NSArray<KRRBFPattern *> *)_patterns targets:(NSArray *)_targets
{
    // 運算原理是每一個 Pattern 都是 Center 的 Candidate，也就是每一個 Pattern 都有可能成為 Center，
    // 這裡開始對每一個中心點做交互運算，例如有 300 筆 Training Patterns，這裡就要跑 300 * 300 = 9 萬次去計算所有 Patterns 彼此之間的距離。
    NSDictionary *_calculatedDistances = [self _calculateCenterDistancesWithPatterns:_patterns];
    
    // 取得 Patterns 和 Candidate Centers 之間運算完的距離資料
    NSDictionary *_centerDistances     = [self _fetchCenterDistancesByResults:_calculatedDistances];
    
    // 計算 Sigma (標準差)
    // 先取出 Patterns 和 Candidate Centers 之間運算完後最大的距離值
    NSNumber *_maxDistance             = [self _fetchMaxDistanceByResults:_calculatedDistances];
    double _sigma                      = [_maxDistance doubleValue] / sqrt([_patterns count]);
    // 用算好的 Sigma 和活化函式來重新運算每一筆 Pattern 和 Centers 之間的距離 (RBF)
    NSMutableArray *_centerRBFDistances = [NSMutableArray new];
    for( KRRBFPattern *_center in _patterns )
    {
        // 先將每一個 Pattern 對同一個 Center 的 RBF Distance 記起來
        NSMutableArray *_rbfDistances = [NSMutableArray new];
        for( KRRBFPattern *_input in _patterns )
        {
            double _distance = [_activeFunction rbf:_center.features x2:_input.features sigma:_sigma];
            [_rbfDistances addObject:[NSNumber numberWithDouble:_distance]]; // phi(j,i)
        }
        // 記錄該 Center 對所有 Patterns 的 RBF 距離, 例如 Center 1 對 10 個 Patterns (這裡在建構 5.17 的 R 矩陣)
        [_centerRBFDistances addObject:_rbfDistances];
    }
    
    // 計算目標輸出值的內積值(Inner Product) t*t'
    double _targetsInnerProduct = [_mathLib sumMatrix:_targets anotherMatrix:_targets];
    // 求每一個 Center 的誤差下降率 (5.25 和下面的函式) 與最大下降誤差值 (下降越大代表誤差減少越多)
    NSMutableArray *_errors     = [NSMutableArray new];
    NSInteger _maxErrorIndex    = -1;
    double _maxErrorValue       = 0.0f;
    // 依序列舉出每一個 Center
    NSInteger _centerNumber     = -1;
    for( NSArray *_centerRBFs in _centerRBFDistances )
    {
        _centerNumber += 1;
        // The formula is following the below steps :
        // RBF values of specified center
        NSArray *_s    = _centerRBFs;
        double _h      = [_mathLib sumMatrix:_s anotherMatrix:_s];
        double _g      = [_mathLib sumMatrix:_s anotherMatrix:_targets] / _h;
        double _error  = _h * (_g * _g) / _targetsInnerProduct;
        [_errors addObject:[NSNumber numberWithDouble:_error]];
        if( _maxErrorIndex < 0 || _error > _maxErrorValue )
        {
            _maxErrorIndex = _centerNumber;
            _maxErrorValue = _error;
        }
    }
    
    // 取出第 1 個挑選到的新中心點
    KRRBFPattern *_newCenter = [_patterns objectAtIndex:_maxErrorIndex];
    _newCenter.isCenter      = YES;

    NSArray *_ss             = [_centerRBFDistances objectAtIndex:_maxErrorIndex];
    double _sumError         = _maxErrorValue;
    
    
}

@end
