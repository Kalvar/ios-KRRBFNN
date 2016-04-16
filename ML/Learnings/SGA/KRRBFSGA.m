//
//  KRRBFSga.m
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/7.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRRBFSGA.h"
#import "KRRBFPattern.h"
#import "KRRBFCenterNet.h"
#import "KRRBFOutputNet.h"
#import "KRMathLib.h"

@interface KRRBFSGA ()

@property (nonatomic, strong) KRMathLib *mathLib;

@end

@implementation KRRBFSGA

+(instancetype)sharedSGA
{
    static dispatch_once_t pred;
    static KRRBFSGA *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRRBFSGA alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _mathLib = [KRMathLib sharedLib];
    }
    return self;
}

// 隨機設定權重
-(NSArray <KRRBFOutputNet *> *)randomWeightsWithCenterCount:(NSInteger)_centerCount targetCount:(NSInteger)_targetCount betweenMin:(double)_minValue max:(double)_maxValue
{
    NSMutableArray *_weights = [NSMutableArray new];
    for( NSInteger _i=0; _i<_targetCount; _i++ )
    {
        KRRBFOutputNet *_outputNet = [[KRRBFOutputNet alloc] init];
        _outputNet.indexKey        = @(_i);
        for( NSInteger _j=0; _j<_centerCount; _j++ )
        {
            [_outputNet addWeight:[NSNumber numberWithDouble:[_mathLib randomDoubleMax:_maxValue min:_minValue]]];
        }
        [_weights addObject:_outputNet];
    }
    return _weights;
}

@end
