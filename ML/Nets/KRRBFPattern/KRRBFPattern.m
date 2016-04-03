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

// 加入該 Pattern 的目標期望值，多分類就依序多執行這裡幾次
-(void)addTarget:(double)_targetValue
{
    [_targets addObject:[NSNumber numberWithDouble:_targetValue]];
}

@end
