//
//  KRRBFNN.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/23.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "KRRBFPattern.h"
#import "KRRBFCenterNet.h"
#import "KRRBFOLS.h"
#import "KRRBFRandom.h"
#import "KRRBFLMS.h"
#import "KRRBFSGA.h"

@class KRRBFTarget;
@class KRRBFNN;

// 中心點選取方法
typedef NS_ENUM(NSInteger, KRRBFNNCenterChoices)
{
    KRRBFNNCenterChoiceOLS = 0,
    KRRBFNNCenterChoiceRandom
};

// 學習方法 (調權重)
typedef NS_ENUM(NSInteger, KRRBFNNLearning)
{
    KRRBFNNLearningLMS = 0,
    KRRBFNNLearningSGA,
    KRRBFNNLearningLMSAndSGA
};

typedef void(^KRRBFNNCompletion)(BOOL success, KRRBFNN *rbfnn);

@interface KRRBFNN : NSObject

@property (nonatomic, strong) NSMutableArray <KRRBFPattern *> *patterns;
@property (nonatomic, strong) NSMutableArray <KRRBFTarget *> *targets;
@property (nonatomic, strong) NSMutableArray <KRRBFCenterNet *> *centers;
@property (nonatomic, strong) NSMutableArray *weights;

// Center choice methods
@property (nonatomic, strong) KRRBFOLS *ols;
@property (nonatomic, strong) KRRBFRandom *random;

// Learning methods
@property (nonatomic, strong) KRRBFLMS *lms;
@property (nonatomic, strong) KRRBFSGA *sga;

// 暫緩使用這 2 個
//@property (nonatomic, assign) KRRBFNNCenterChoices centerChoiceMethod;
//@property (nonatomic, assign) KRRBFNNLearning learningMethod;

+(instancetype)sharedNetwork;
-(instancetype)init;

-(void)addPattern:(KRRBFPattern *)_pattern;
-(void)addPatterns:(NSArray <KRRBFPattern *> *)_samples;

-(NSArray <KRRBFCenterNet *> *)pickingCentersByOLSWithTolerance:(double)_tolerance setToCenters:(BOOL)_setToCenters;
-(NSArray <KRRBFCenterNet *> *)pickingCentersByRandomWithLimitCount:(NSInteger)_limitCount setToCenters:(BOOL)_setToCenters;
-(void)trainingByLMSWithCompletion:(KRRBFNNCompletion)_completion;
-(void)trainingBySGAWithCompletion:(KRRBFNNCompletion)_completion;

@end
