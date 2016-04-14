//
//  KRRBFPassed.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/12.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFPassed.h"
#import "KRRBFCenterNet.h"
#import "KRRBFOutputNet.h"

@interface KRRBFPassed ()

@property (nonatomic, weak) NSCoder *coder;

@end

@implementation KRRBFPassed (Coding)

-(void)_encodeObject:(id)_object forKey:(NSString *)_key
{
    if( nil != _object )
    {
        [self.coder encodeObject:_object forKey:_key];
    }
}

-(id)_decodeForKey:(NSString *)_key
{
    return [self.coder decodeObjectForKey:_key];
}

@end

@implementation KRRBFPassed

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _centers = [NSMutableArray new];
        _weights = [NSMutableArray new];
    }
    return self;
}

#pragma --mark NSCoding
-(void)encodeWithCoder:(NSCoder *)aCoder
{
    self.coder = aCoder;
    [self _encodeObject:_centers forKey:@"centers"];
    [self _encodeObject:_weights forKey:@"weights"];
}

-(instancetype)initWithCoder:(NSCoder *)aDecoder
{
    self = [super init];
    if(self)
    {
        self.coder = aDecoder;
        _centers   = [self _decodeForKey:@"centers"];
        _weights   = [self _decodeForKey:@"weights"];
    }
    return self;
}

@end
