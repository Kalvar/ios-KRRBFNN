//
//  KRRBFOutput.m
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFOutputNet.h"

@implementation KRRBFOutputNet

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _weights      = [NSMutableArray new];
        _targetValue  = 0.0f;
        _outputValue  = 0.0f;
        
        self.indexKey = nil;
    }
    return self;
}

-(void)removeAllWeights
{
    if( nil != _weights && [_weights count] > 0 )
    {
        [_weights removeAllObjects];
    }
}

-(void)addWeightsFromArray:(NSArray *)_outputWeights
{
    [self removeAllWeights];
    if( nil != _outputWeights )
    {
        [_weights addObjectsFromArray:_outputWeights];
        //_weights = [[NSMutableArray alloc] initWithArray:_outputWeights copyItems:YES];
    }
}

-(double)outputWithRBFValues:(NSArray *)_rbfValues
{
    double _sum      = 0.0f;
    NSInteger _index = -1;
    for( NSNumber *_netWeight in _weights )
    {
        _index              += 1;
        NSNumber *_rbfValue  = [_rbfValues objectAtIndex:_index];
        _sum                += ( [_rbfValue doubleValue] * [_netWeight doubleValue] );
    }
    _outputValue = _sum;
    return _sum;
}

#pragma --mark Getters
-(double)outputError
{
    return _outputValue - _targetValue;
}

@end
