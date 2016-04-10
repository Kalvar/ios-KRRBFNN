//
//  KRRBFSga.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/7.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFSGA.h"

@implementation KRRBFSGA

+(instancetype)sharedSGA
{
    static dispatch_once_t pred;
    static KRRBFSGA *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFSGA alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

@end
