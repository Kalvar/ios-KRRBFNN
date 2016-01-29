//
//  KRRBFOLS.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/1/29.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRRBFOLS : NSObject

// Chose centers by algorithm calculated
@property (nonatomic, strong) NSMutableArray *centers;

+(instancetype)sharedOLS;
-(instancetype)init;

@end
