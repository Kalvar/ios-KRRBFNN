//
//  KRRBFPattern.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFPattern.h"

@implementation KRRBFPattern

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _targets = [NSMutableArray new];
    }
    return self;
}

-(void)addFeature:(double)_featureValue
{
    [self.features addObject:[NSNumber numberWithDouble:_featureValue]];
}

// 加入該 Pattern 的目標期望值，多分類就依序多執行這裡幾次
-(void)addTarget:(double)_targetValue
{
    [_targets addObject:[NSNumber numberWithDouble:_targetValue]];
}

#pragma --mark NSCopying
-(instancetype)copyWithZone:(NSZone *)zone
{
    KRRBFPattern *_p = [[KRRBFPattern alloc] init];
    _p.features      = [[NSMutableArray alloc] initWithArray:self.features copyItems:YES]; // copyItems is YES that means whole deeply copying
    _p.indexKey      = [self.indexKey copy];
    _p.targets       = [[NSMutableArray alloc] initWithArray:_targets copyItems:YES];
    return _p;
}

@end
