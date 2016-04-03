//
//  KRRBFCandidateCenter.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFCandidateCenter.h"

@implementation KRRBFCandidateCenter

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        
    }
    return self;
}

-(void)recordPatternIndex:(NSNumber *)_pIndex centerIndex:(NSNumber *)_cIndex distance:(double)_diffDistance
{
    _patternIndex = [_pIndex copy];
    _centerIndex  = [_cIndex copy];
    _distance     = _diffDistance;
}

@end
