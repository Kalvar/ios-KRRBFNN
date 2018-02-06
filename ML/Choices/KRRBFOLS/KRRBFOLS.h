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

// tolerance 在 OLS 裡是誤差下降率的容忍度，例如: 設 0.8 代表最大誤差下降率必須達到 80% 才算完成整個挑選的過程，把 tolerance 看成正確率的話也是對的，
// 正確率越高，誤差範圍越小 (最大誤差下降率越大)，所以，正確率越趨近於 1.0，則預測效果越好，隱藏層神經元就越多。
@property (nonatomic, assign) double tolerance;
// 最多選幾顆當 Center
@property (nonatomic, assign) NSInteger maxPick;

+(instancetype)sharedOLS;
-(instancetype)init;

-(NSArray <KRRBFCenterNet *> *)chooseWithPatterns:(NSArray<KRRBFPattern *> *)_patterns targets:(NSArray<KRRBFTarget *> *)_targets;

@end
