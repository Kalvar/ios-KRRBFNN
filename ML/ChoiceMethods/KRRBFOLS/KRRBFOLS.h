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

@interface KRRBFOLS : NSObject

// Chose centers by algorithm calculated
@property (nonatomic, strong) NSMutableArray *centers;
// 正確率 (容忍值 ; 收斂誤差 ; RBFNN 正確率越大代表誤差越小 ?)
@property (nonatomic, assign) double tolerance;

+(instancetype)sharedOLS;
-(instancetype)init;

-(NSArray *)olsWithPatterns:(NSArray<KRRBFPattern *> *)_patterns targets:(NSArray<KRRBFTarget *> *)_targets;

-(void)testOls;

@end
