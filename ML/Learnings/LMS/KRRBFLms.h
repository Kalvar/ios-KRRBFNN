//
//  KRRBFLms.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/7.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRRBFPattern;
@class KRRBFCenterNet;
@class KRRBFOutputNet;
@class KRRBFTarget;

@interface KRRBFLMS : NSObject

+(instancetype)sharedLMS;
-(instancetype)init;

-(NSArray <KRRBFOutputNet *> *)outputWeightsWithCenters:(NSArray <KRRBFCenterNet *> *)_centers patterns:(NSArray <KRRBFPattern *> *)_patterns targets:(NSArray <KRRBFTarget *> *)_targets;

@end
