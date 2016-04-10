//
//  KRRBFHiddenLayer.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFHiddenLayer.h"
#import "KRRBFCenterNet.h"

@implementation KRRBFHiddenLayer

+(instancetype)sharedLayer
{
    static dispatch_once_t pred;
    static KRRBFHiddenLayer *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFHiddenLayer alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _nets = [NSMutableArray new];
    }
    return self;
}

-(void)removeAllCenters
{
    if( nil != _nets )
    {
        [_nets removeAllObjects];
    }
}

-(void)addCentersFromArray:(NSArray<KRRBFCenterNet *> *)choseCenters
{
    [self removeAllCenters];
    [_nets addObjectsFromArray:choseCenters];
}

@end
