//
//  KRCenterChoose.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRRBFCenterChoose : NSObject

@property (nonatomic, strong) NSMutableArray *patterns;

+(instancetype)sharedChoose;
-(instancetype)init;

@end
