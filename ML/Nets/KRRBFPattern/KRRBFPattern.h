//
//  KRRBFPattern.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFNet.h"

@interface KRRBFPattern : KRRBFNet

@property (nonatomic, strong) NSMutableArray *targets; // 本 Pattern 的目標輸出值

-(instancetype)init;
-(void)addFeature:(double)_featureValue;
-(void)addTarget:(double)_targetValue;

@end
