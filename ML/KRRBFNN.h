//
//  KRRBFNN.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "KRRBFHiddenLayer.h"
#import "KRRBFOutputLayer.h"

#import "KRRBFPattern.h"
#import "KRRBFCenterNet.h"
#import "KRRBFOutputNet.h"

#import "KRRBFOLS.h"
#import "KRRBFRandom.h"
#import "KRRBFLMS.h"
#import "KRRBFSGA.h"

@class KRRBFTarget;
@class KRRBFNN;

typedef void(^KRRBFNNCompletion)(BOOL success, KRRBFNN *rbfnn);

@interface KRRBFNN : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFPattern *> *patterns;
@property (nonatomic, strong) NSMutableArray <KRRBFTarget *> *targets;

@property (nonatomic, strong, readonly) NSArray <KRRBFCenterNet *> *centers;
@property (nonatomic, strong, readonly) NSArray <KRRBFOutputNet *> *weights;

@property (nonatomic, strong) KRRBFHiddenLayer *hiddenLayer;
@property (nonatomic, strong) KRRBFOutputLayer *outputLayer;

// Center choice methods
@property (nonatomic, strong) KRRBFOLS *ols;
@property (nonatomic, strong) KRRBFRandom *random;

// Learning methods
@property (nonatomic, strong) KRRBFLMS *lms;
@property (nonatomic, strong) KRRBFSGA *sga;

+(instancetype)sharedNetwork;
-(instancetype)init;

-(void)addPattern:(KRRBFPattern *)_pattern;
-(void)addPatterns:(NSArray <KRRBFPattern *> *)_samples;

-(NSArray <KRRBFCenterNet *> *)pickingCentersByOLSWithTolerance:(double)_tolerance setToCenters:(BOOL)_setToCenters;
-(NSArray <KRRBFCenterNet *> *)pickingCentersByRandomWithLimitCount:(NSInteger)_limitCount setToCenters:(BOOL)_setToCenters;
-(void)trainingByLMSWithCompletion:(KRRBFNNCompletion)_completion;
-(void)trainingBySGAWithCompletion:(KRRBFNNCompletion)_completion;

@end
