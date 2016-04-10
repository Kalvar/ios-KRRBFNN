//
//  KRRBFNet.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/3/30.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFNet.h"

@implementation KRRBFNet

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _features = [NSMutableArray new];
        _indexKey = nil;
        _bias     = 0.0f;
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

#pragma --mark NSCopying
-(instancetype)copyWithZone:(NSZone *)zone
{
    KRRBFNet *_net = [[KRRBFNet alloc] init];
    _net.features  = [[NSMutableArray alloc] initWithArray:_features copyItems:YES]; // Whole deeply copying
    _net.indexKey  = _indexKey ? [_indexKey copy] : nil;
    _net.bias      = _bias;
    return _net;
}

@end
