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

static NSString *kKRRBFOLSCalculatedDistancesKey   = @"distances";
static NSString *kKRRBFOLSCalculatedMaxDistanceKey = @"maxDistance";

@interface KRRBFOLS ()

@property (nonatomic, strong) KRRBFActiviation *activeFunction;

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
            [_candidateCenter recordPatternIndex:_pattern.index centerIndex:_candidate.index distance:_distance];
            [_centerDistances addObject:_candidateCenter];
            if( _distance > _maxDistance )
            {
                _maxDistance = _distance;
            }
        }
        [_withCenters setObject:_centerDistances forKey:_pattern.index];
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
    }
    return self;
}

-(void)olsWithPatterns:(NSArray *)_patterns
{
    // 運算原理是每一個 Pattern 都是 Center 的 Candidate，也就是每一個 Pattern 都有可能成為 Center，
    // 這裡開始對每一個中心點做交互運算，例如有 300 筆 Training Patterns，這裡就要跑 300 * 300 = 9 萬次去計算所有 Patterns 彼此之間的距離。
    NSDictionary *_calculatedDistances = [self _calculateCenterDistancesWithPatterns:_patterns];
    
    // 取得 Patterns 和 Candidate Centers 之間運算完的距離資料
    NSDictionary *_centerDistances     = [self _fetchCenterDistancesByResults:_calculatedDistances];
    
    // 計算 Sigma
    // 先取出 Patterns 和 Candidiate Centers 之間運算完後最大的距離值
    NSNumber *_maxDistance             = [self _fetchMaxDistanceByResults:_calculatedDistances];
    double _sigma                      = [_maxDistance doubleValue] / sqrt([_patterns count]);
    
    // 用算好的 Sigma 和活化函式來重新運算每一個 Patterns 和 Centers 之間的距離 (RBF)
    NSMutableDictionary *_rbfDistances = [NSMutableDictionary new];
    for( KRRBFPattern *_pattern in _patterns )
    {
        for( KRRBFPattern *_candidate in _patterns )
        {
            double _distance = [self.activeFunction rbf:_pattern.features x2:_candidate.features sigma:_sigma];
        }
    }
    
}

@end
