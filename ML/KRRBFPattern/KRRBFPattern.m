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
        _features    = [NSMutableArray new];
        _indexKey = nil;
        _isCenter    = NO;
    }
    return self;
}

-(void)addFeatures:(NSArray *)_f
{
    if( nil != _f && [_f count] > 0 )
    {
        [_features addObjectsFromArray:_f];
    }
}

-(void)setIndexNumberAtIndex:(NSInteger)_index
{
    _indexKey = [NSNumber numberWithInteger:_index];
}

@end
