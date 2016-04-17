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
        _mathLib            = [KRMathLib sharedLib];
        _weightLearningRate = 1.0f;
        _centerLearningRate = 1.0f;
        _sigmaLearningRate  = 1.0f;
        
        // 這裡刻意使用 memory reference 來讓 centers, weights, patterns 都會在更新時，一併連動外部的參數一起修改，如此能更有效率的運算，
        // 如果這樣的記憶體連動方法不妥，以後再於重構時進行優化。
        _centers            = nil;
        _weights            = nil;
        _patterns           = nil;
    }
    return self;
}

#pragma --mark Weights & Output Nets
// 隨機設定權重
-(NSArray <KRRBFOutputNet *> *)randomWeightsWithCenterCount:(NSInteger)_centerCount targetCount:(NSInteger)_targetCount betweenMin:(double)_minValue max:(double)_maxValue
{
    NSMutableArray *_randomWeights = [NSMutableArray new];
    for( NSInteger _i=0; _i<_targetCount; _i++ )
    {
        KRRBFOutputNet *_outputNet = [[KRRBFOutputNet alloc] init];
        _outputNet.indexKey        = @(_i);
        for( NSInteger _j=0; _j<_centerCount; _j++ )
        {
            // If maxValue && minValue are 0.0f that weight also be zero.
            [_outputNet addWeight:[NSNumber numberWithDouble:[_mathLib randomDoubleMax:_maxValue min:_minValue]]];
        }
        [_randomWeights addObject:_outputNet];
    }
    return _randomWeights;
}

// 設定全零的權重
-(NSArray <KRRBFOutputNet *> *)zeroWeightsWithCenterCount:(NSInteger)_centerCount targetCount:(NSInteger)_targetCount
{
    return [self randomWeightsWithCenterCount:_centerCount targetCount:_targetCount betweenMin:0.0f max:0.0f];
}

#pragma --mark Update Methods
// To use reference memory mechanism
-(void)updateWeights
{
#warning  TODO:
    
}

-(void)updateCenters
{
#warning  TODO:
    
}

-(void)updateSigmas
{
#warning  TODO:
    
}

#pragma --mark Free Memory
-(void)freeReferences
{
    // Could set nil to release the weak reference then help ARC to early recycle that memory.
    _centers  = nil;
    _weights  = nil;
    _patterns = nil;
}

#pragma --mark Setters
-(void)setLearningRate:(double)_rate
{
    _weightLearningRate = _rate;
    _centerLearningRate = _rate;
    _sigmaLearningRate  = _rate;
    _learningRate       = _rate;
}

-(void)dealloc
{
    NSLog(@"KRRBFSGA is dealloced");
}

@end
