//
//  KRRBFCenter.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import "KRRBFNet.h"

@interface KRRBFCenterNet : KRRBFNet

-(instancetype)initWithFeatures:(NSArray *)_features;
-(instancetype)init;
-(void)copyWithFeatures:(NSArray *)_features;

@end
