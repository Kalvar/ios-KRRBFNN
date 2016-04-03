//
//  KRRBFCandidateCenter.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRRBFCandidateCenter : NSObject

// 跟哪一個 Pattern
@property (nonatomic, strong) NSNumber *patternIndex;
// 自己是哪一個 Center
@property (nonatomic, strong) NSNumber *centerIndex;
// Pattern 跟 Center 之間的距離
@property (nonatomic, assign) double distance;

-(instancetype)init;
-(void)recordPatternIndex:(NSNumber *)_pIndex centerIndex:(NSNumber *)_cIndex distance:(double)_diffDistance;

@end
