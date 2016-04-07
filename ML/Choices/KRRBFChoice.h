//
//  KRCenterChoice.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "KRRBFOLS.h"
#import "KRRBFRandom.h"

// 中心點選取方法
@interface KRRBFChoice : NSObject

@property (nonatomic, strong) NSMutableArray *patterns;
@property (nonatomic, strong) KRRBFOLS *ols;
@property (nonatomic, strong) KRRBFRandom *random;

+(instancetype)sharedChoice;
-(instancetype)init;

@end
