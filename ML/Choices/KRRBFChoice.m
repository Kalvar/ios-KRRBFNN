//
//  KRCenterChoice.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFChoice.h"

@interface KRRBFChoice ()

@end

@implementation KRRBFChoice

+(instancetype)sharedChoice
{
    static dispatch_once_t pred;
    static KRRBFChoice *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFChoice alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _patterns = [NSMutableArray new];
        /*
         * @ TODO:
         *   - 1. 待補上 OLS + SGA
         *   - 2. 待補上 隨機選取法 + SGA + KMeans
         */
    }
    return self;
}

#pragma --mark Getters
-(KRRBFOLS *)ols
{
    if( nil == _ols )
    {
        _ols = [KRRBFOLS sharedOLS];
    }
    return _ols;
}

-(KRRBFRandom *)random
{
    if ( nil == _random )
    {
        _random = [KRRBFRandom sharedRandom];
    }
    return _random;
}

@end
