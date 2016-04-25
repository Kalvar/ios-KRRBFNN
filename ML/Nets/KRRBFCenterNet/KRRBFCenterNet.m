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
        _sigma = -1.0f; // Default -1.0f means nothing.
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

-(void)addFeaturesFromArray:(NSArray *)_features
{
    if( self.features && _features )
    {
        [self.features removeAllObjects];
        [self.features addObjectsFromArray:_features];
    }
}

#pragma --mark NSCopying
-(instancetype)copyWithZone:(NSZone *)zone
{
    KRRBFCenterNet *_net = [[[self class] alloc] init];
    [_net setFeatures:[[NSMutableArray alloc] initWithArray:self.features copyItems:YES]];
    [_net setIndexKey:[self.indexKey copy]];
    [_net setSigma:_sigma];
    [_net setRbfValue:_rbfValue];
    return _net;
}

#pragma --mark NSCoding
-(void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder; // Inherited from parent class.
    [self encodeObject:self.features forKey:@"features"];
    [self encodeObject:self.indexKey forKey:@"indexKey"];
    [self encodeObject:[NSNumber numberWithDouble:_sigma] forKey:@"sigma"];
    [self encodeObject:[NSNumber numberWithDouble:_rbfValue] forKey:@"rbfValue"];
}

-(instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder    = aDecoder;
        self.features = [self decodeForKey:@"features"];
        self.indexKey = [self decodeForKey:@"indexKey"];
        _sigma        = [[self decodeForKey:@"sigma"] doubleValue];
        _rbfValue     = [[self decodeForKey:@"rbfValue"] doubleValue];
    }
    return self;
}

@end
