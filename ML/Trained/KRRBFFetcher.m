//
//  KRRBFFetcher.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/14.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFFetcher.h"
#import "KRRBFNN.h"

@implementation KRRBFFetcher

+(instancetype)sharedFetcher
{
    static dispatch_once_t pred;
    static KRRBFFetcher *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFFetcher alloc] init];
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

-(void)save:(KRRBFPassed *)object forKey:(NSString *)key
{
    if( object && key )
    {
        [[NSUserDefaults standardUserDefaults] setObject:[NSKeyedArchiver archivedDataWithRootObject:object] forKey:key];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

-(void)removeForKey:(NSString *)key
{
    if( key )
    {
        [[NSUserDefaults standardUserDefaults] removeObjectForKey:key];
        [[NSUserDefaults standardUserDefaults] synchronize];
    }
}

// Fetching saved and trained network.
-(KRRBFPassed *)objectForKey:(NSString *)key
{
    if( key )
    {
        NSData *_objectData = [[NSUserDefaults standardUserDefaults] valueForKey:key];
        return _objectData ? [NSKeyedUnarchiver unarchiveObjectWithData:_objectData] : nil;
    }
    return nil;
}

@end
