//
//  KRRBFRandom.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/3.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@class KRRBFCenterNet;
@class KRRBFPattern;

@interface KRRBFRandom : NSObject

+(instancetype)sharedRandom;
-(instancetype)init;

// If pickNumber is <= 0 that means to use automatic random picking.
-(NSArray <KRRBFCenterNet *> *)chooseWithPatterns:(NSArray<KRRBFPattern *> *)_patterns pickNumber:(NSInteger)_pickNumber;
// Automatic choose centers with system random picking.
-(NSArray <KRRBFCenterNet *> *)chooseAutomaticallyWithPatterns:(NSArray<KRRBFPattern *> *)_patterns;

-(void)removeCaches;

@end
