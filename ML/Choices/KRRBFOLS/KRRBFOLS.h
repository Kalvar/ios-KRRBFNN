//
//  KRRBFOLS.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRRBFPattern;
@class KRRBFTarget;
@class KRRBFCenterNet;

@interface KRRBFOLS : NSObject

// Chose centers by algorithm calculated
@property (nonatomic, strong) NSMutableArray *centers;
// 正確率 (RBFNN 使用的收斂誤差是正確率，即 1.0f - 正確率 = 一般類神經網路的收斂誤差)
// 正確率越趨近於 1.0，則預測效果越好，隱藏層神經元就越多
@property (nonatomic, assign) double tolerance;
// 最多選幾顆當 Center
@property (nonatomic, assign) NSInteger maxPick;

+(instancetype)sharedOLS;
-(instancetype)init;

-(NSArray <KRRBFCenterNet *> *)chooseWithPatterns:(NSArray<KRRBFPattern *> *)_patterns targets:(NSArray<KRRBFTarget *> *)_targets;

@end
