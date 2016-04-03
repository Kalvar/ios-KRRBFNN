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
        _isCenter = NO;
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

@end
