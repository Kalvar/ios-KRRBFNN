//
//  KRRBFRandom.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/3.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFRandom.h"
#import "KRRBFCenterNet.h"
#import "KRRBFPattern.h"

@interface KRRBFRandom ()

// 將算好的隨機陣列快取起來
@property (nonatomic, strong) NSMutableArray <KRRBFPattern *> *cacheRandoms;

@end

@implementation KRRBFRandom (RandomAlgorithm)

// 用洗牌法進行亂數取出
-(NSMutableArray <KRRBFPattern *> *)randomAlgorithm:(NSArray<KRRBFPattern *> *)_patterns
{
    NSMutableArray *_samples = [_patterns mutableCopy];
    NSInteger _totalLength   = [_patterns count];
    for( NSInteger _i=0; _i<_totalLength; _i++ )
    {
        NSInteger _random1 = ( arc4random() % _totalLength );
        NSInteger _random2 = ( arc4random() % _totalLength );
        // 如果亂數重複，則用範本數長度減去亂數值
        if( _random1 == _random2 )
        {
            _random2 = _totalLength - _random2;
        }
        // 進行陣列交換
        // 先取出 random1 位置的 Object
        KRRBFPattern *_temp = [_samples objectAtIndex:_random1];
        // 再將 random2 位置的 Object 塞回去 random1 位置
        [_samples replaceObjectAtIndex:_random1 withObject:[_samples objectAtIndex:_random2]];
        // 最後將剛才取出的 random1 Object 塞回去 random2 即可
        [_samples replaceObjectAtIndex:_random2 withObject:_temp];
    }
    return _samples;
}

@end

@implementation KRRBFRandom

+(instancetype)sharedRandom
{
    static dispatch_once_t pred;
    static KRRBFRandom *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFRandom alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _cacheRandoms = [NSMutableArray new];
    }
    return self;
}

-(NSArray <KRRBFCenterNet *> *)chooseWithPatterns:(NSArray<KRRBFPattern *> *)_patterns pickNumber:(NSInteger)_pickNumber
{
    NSInteger _totalLength = [_patterns count];
    // 先正規化 pickNumber 以避免 <= 0 和 > _totalLength 的狀況
    if( _pickNumber <= 0 )
    {
        _pickNumber = ( arc4random() % _totalLength );
    }
    else if( _pickNumber > _totalLength )
    {
        _pickNumber = _totalLength;
    }
    
    if( nil == _cacheRandoms || [_cacheRandoms count] < 1 )
    {
        _cacheRandoms = [self randomAlgorithm:_patterns];
    }
    
    NSMutableArray <KRRBFCenterNet *> *_picks = [NSMutableArray new];
    for( NSInteger _j=0; _j<_pickNumber; _j++ )
    {
        KRRBFCenterNet *_centerNet = [[KRRBFCenterNet alloc] initWithFeatures:[_cacheRandoms objectAtIndex:_j].features];
        _centerNet.indexKey        = @(_j);
        [_picks addObject:_centerNet];
    }
    return _picks;
}

-(NSArray <KRRBFCenterNet *> *)chooseAutomaticallyWithPatterns:(NSArray<KRRBFPattern *> *)_patterns
{
    return [self chooseWithPatterns:_patterns pickNumber:-1];
}

-(void)removeCaches
{
    if( nil != _cacheRandoms )
    {
        [_cacheRandoms removeAllObjects];
    }
}

/*
-(void)dealloc
{
    NSLog(@"KRRBFRandom is dealloced");
}
 */

@end
