//
//  KRRBFRandom.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/3.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFRandom.h"

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
        
    }
    return self;
}

@end
