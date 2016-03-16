//
//  KRRBFPattern.h
//  RBFNN
//
//  Created by Kalvar Lin on 2015/12/25.
//  Copyright © 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRRBFPattern : NSObject

@property (nonatomic, strong) NSMutableArray *features;
@property (nonatomic, strong) NSNumber *indexKey;
@property (nonatomic, assign) BOOL isCenter;

-(instancetype)init;
-(void)addFeatures:(NSArray *)_f;
-(void)setIndexNumberAtIndex:(NSInteger)_index;

@end
