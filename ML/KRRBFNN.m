//
//  KRRBFNN.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFNN.h"

@implementation KRRBFNN

+(instancetype)sharedNetwork
{
    static dispatch_once_t pred;
    static KRRBFNN *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFNN alloc] init];
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
