//
//  KRRBFSga.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/7.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRRBFOutputNet;

@interface KRRBFSGA : NSObject

+(instancetype)sharedSGA;
-(instancetype)init;

-(NSArray <KRRBFOutputNet *> *)randomWeightsWithCenterCount:(NSInteger)_centerCount targetCount:(NSInteger)_targetCount betweenMin:(double)_minValue max:(double)_maxValue;

@end
