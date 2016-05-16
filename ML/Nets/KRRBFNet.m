//
//  KRRBFNet.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/3/30.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFNet.h"

@implementation KRRBFNet (NSCoding)

-(void)encodeObject:(id)_object forKey:(NSString *)_key
{
    if( nil != _object )
    {
        [self.coder encodeObject:_object forKey:_key];
    }
}

-(id)decodeForKey:(NSString *)_key
{
    return [self.coder decodeObjectForKey:_key];
}

@end

@implementation KRRBFNet

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _features = [NSMutableArray new];
        _indexKey = nil;
        _bias     = 0.0f;
    }
    return self;
}

-(void)addFeatures:(NSArray *)_f
{
    if( nil != _f && [_f count] > 0 )
    {
        [_features addObjectsFromArray:_f];
    }
}

#pragma --mark Getter
-(NSNumber *)indexKey
{
    if( nil == _indexKey )
    {
        // To use milliseconds to be default indexKey if it is nil.
        _indexKey = @([[NSDate date] timeIntervalSince1970] * 1000);
    }
    return _indexKey;
}

#pragma --mark NSCopying
-(instancetype)copyWithZone:(NSZone *)zone
{
    KRRBFNet *_net = [[KRRBFNet alloc] init];
    _net.features  = [[NSMutableArray alloc] initWithArray:_features copyItems:YES]; // Whole deeply copying
    _net.indexKey  = [_indexKey copy];
    _net.bias      = _bias;
    return _net;
}

#pragma --mark NSCoding
-(void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
    [self encodeObject:_features forKey:@"features"];
    //[self encodeObject:_indexKey forKey:@"indexKey"];
}

-(instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder = aDecoder;
        _features   = [self decodeForKey:@"features"];
        //_indexKey   = [self decodeForKey:@"indexKey"];
    }
    return self;
}

@end
