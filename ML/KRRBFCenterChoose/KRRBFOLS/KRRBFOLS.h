//
//  KRRBFOLS.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRRBFOLS : NSObject

// Chose centers by algorithm calculated
@property (nonatomic, strong) NSMutableArray *centers;
// 正確率 (容忍值 ; 收斂誤差)
@property (nonatomic, assign) double tolerance;

+(instancetype)sharedOLS;
-(instancetype)init;

@end
