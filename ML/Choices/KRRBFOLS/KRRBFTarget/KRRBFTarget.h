//
//  KRRBFTarget.h
//  RBFNN
//
//  Created by Kalvar Lin on 2016/3/16.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

// 用於儲存要在 OLS 方法裡計算的所有 Patterns 同維度期望輸出值
@interface KRRBFTarget : NSObject

@property (nonatomic, assign) NSInteger sequenceIndex;        // 是哪一個位置的期望輸出
@property (nonatomic, strong) NSMutableArray *sameSequences;  // 相同位置(維度)的期望輸出

-(instancetype)init;

@end
