//
//  KRRBFFetcher.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/4/14.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "KRRBFPassed.h"

@interface KRRBFFetcher : NSObject

+(instancetype)sharedFetcher;
-(instancetype)init;

-(void)save:(KRRBFPassed *)object forKey:(NSString *)key;  // Saving network information with key.
-(void)removeForKey:(NSString *)key;                      // Removes saved network information with key.
-(KRRBFPassed *)objectForKey:(NSString *)key;             // Fetching saved network with key.

@end
