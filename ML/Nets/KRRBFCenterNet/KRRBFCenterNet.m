//
//  KRRBFCenter.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFCenterNet.h"

@implementation KRRBFCenterNet

-(instancetype)initWithFeatures:(NSArray *)_features
{
    self = [super init];
    if( self )
    {
        [self copyWithFeatures:_features];
    }
    return self;
}

-(instancetype)init
{
    self = [self initWithFeatures:nil];
    if( self )
    {
        
    }
    return self;
}

// Deep copy with another array.
-(void)copyWithFeatures:(NSArray *)_features
{
    if( _features )
    {
        self.features = [[NSMutableArray alloc] initWithArray:_features copyItems:YES];
    }
}

@end
