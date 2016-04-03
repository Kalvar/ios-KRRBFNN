//
//  KRRBFTarget.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/3/16.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFTarget.h"

@implementation KRRBFTarget

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _sequenceIndex = 0;
        _sameSequences = [NSMutableArray new];
    }
    return self;
}

@end
