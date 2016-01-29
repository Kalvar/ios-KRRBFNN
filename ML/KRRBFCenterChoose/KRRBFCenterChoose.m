//
//  KRCenterChoose.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFCenterChoose.h"
#import "KRRBFOLS.h"

@interface KRRBFCenterChoose ()

@property (nonatomic, strong) KRRBFOLS *rbfOLS;

@end

@implementation KRRBFCenterChoose

+(instancetype)sharedChoose
{
    static dispatch_once_t pred;
    static KRRBFCenterChoose *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFCenterChoose alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _patterns = [NSMutableArray new];
        _rbfOLS   = [KRRBFOLS sharedOLS];
    }
    return self;
}



@end
