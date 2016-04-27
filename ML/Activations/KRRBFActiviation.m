//
//  KRRBFActiviation.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFActiviation.h"

@implementation KRRBFActiviation

+(instancetype)sharedActiviation
{
    static dispatch_once_t pred;
    static KRRBFActiviation *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFActiviation alloc] init];
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

-(double)euclidean:(NSArray *)_x1 x2:(NSArray *)_x2
{
    NSInteger _index = 0;
    double _sum      = 0.0f;
    for( NSNumber *_x in _x1 )
    {
        _sum        += powf([_x doubleValue] - [[_x2 objectAtIndex:_index] doubleValue], 2);
        ++_index;
    }
    // 累加完距離後直接開根號
    return (_index > 0) ? sqrtf(_sum) : _sum;
}

-(double)rbf:(NSArray *)_x1 x2:(NSArray *)_x2 sigma:(float)_sigma
{
    double _sum      = 0.0f;
    NSInteger _index = 0;
    for( NSNumber *_value in _x1 )
    {
        // Formula : s = s + ( v1[i] - v2[i] )^2
        double _v  = [_value doubleValue] - [[_x2 objectAtIndex:_index] doubleValue];
        _sum      += ( _v * _v );
        ++_index;
    }
    // Formula : exp^( -s / ( 2.0f * sigma * sigma ) )
    return pow(M_E, ((-_sum) / ( 2.0f * _sigma * _sigma )));
}


@end
