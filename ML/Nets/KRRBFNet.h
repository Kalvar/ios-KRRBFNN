//
//  KRRBFNet.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/3/30.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRRBFNet : NSObject

@property (nonatomic, strong) NSMutableArray *features;
@property (nonatomic, strong) NSNumber *indexKey;
@property (nonatomic, assign) BOOL isCenter;

-(instancetype)init;
-(void)addFeatures:(NSArray *)_f;

@end
