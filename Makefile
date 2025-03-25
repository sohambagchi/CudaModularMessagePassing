NVCC = $(shell which nvcc)
NVCC_FLAGS := -std=c++14 -g -Xcompiler -O0 -Xcicc -O2 -arch=sm_87
NVCC_FLAGS += --expt-relaxed-constexpr
NVCC_FLAGS += -arch=sm_87

OUTPUT = output/MP
# PTX = MP_PTX

SRCS := message_passing.cu

VAR_SCOPES_X = X_CTA X_DEV
VAR_SCOPES_Y = Y_CTA Y_DEV Y_SYS

PRODUCER_FENCES = PRODUCER_FENCE_ACQ_REL PRODUCER_FENCE_SC PRODUCER_NO_FENCE
PRODUCER_FENCE_SCOPES = PRODUCER_FENCE_SCOPE_CTA PRODUCER_FENCE_SCOPE_DEV PRODUCER_FENCE_SCOPE_SYS

CONSUMER_FENCES = CONSUMER_FENCE_ACQ_REL CONSUMER_FENCE_SC CONSUMER_NO_FENCE
CONSUMER_FENCE_SCOPES = CONSUMER_FENCE_SCOPE_CTA CONSUMER_FENCE_SCOPE_DEV CONSUMER_FENCE_SCOPE_SYS

ORDERS = RLX_RLX REL_ACQ RLX_ACQ REL_RLX



# all: $(foreach x_scope,$(VAR_SCOPES_X),$(foreach y_scope,$(VAR_SCOPES_Y),$(foreach producer_fence,$(PRODUCER_FENCES),$(foreach consumer_fence,$(CONSUMER_FENCES),$(foreach order,$(ORDERS),$(OUTPUT)_$(x_scope)_$(y_scope)_$(producer_fence)_$(consumer_fence)_$(order).out)))))

all: $(foreach x_scope,$(VAR_SCOPES_X),$(foreach y_scope,$(VAR_SCOPES_Y),$(foreach producer_fence,$(PRODUCER_FENCES),$(foreach producer_scope,$(PRODUCER_FENCE_SCOPES),$(foreach consumer_fence,$(CONSUMER_FENCES),$(foreach consumer_scope,$(CONSUMER_FENCE_SCOPES),$(foreach order,$(ORDERS),$(OUTPUT)-$(x_scope)-$(y_scope)-$(producer_fence)-$(producer_scope)-$(consumer_fence)-$(consumer_scope)-$(order).out)))))))


# all:  $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_rel_$(scope)_$(size)_$(buf).out))) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_rlx_$(scope)_$(size)_$(buf).out))) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_no_acq_rel_$(scope)_$(size)_$(buf).out))) $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_no_acq_rlx_$(scope)_$(size)_$(buf).out)))

# flag-rel: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_rel_$(scope)_$(size)_$(buf).out)))

# flag-rlx: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_rlx_$(scope)_$(size)_$(buf).out)))

# no-acq-flag-rel: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_no_acq_rel_$(scope)_$(size)_$(buf).out)))

# no-acq-flag-rlx: $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(OUTPUT)_no_acq_rlx_$(scope)_$(size)_$(buf).out)))

define make_target
$(OUTPUT)-$(1)-$(2)-$(3)-$(4)-$(5)-$(6)-$(7).out: $(SOURCES)
	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -D$(4) -D$(5) -D$(6) -D$(7) -o $$@ $$(SRCS)
endef

# define make_target_rel
# $(OUTPUT)_rel_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
# 	# $$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -DP_H_FLAG_STORE_ORDER_REL  -o $$@ $$(SRC)
# endef

# define make_target_rlx
# $(OUTPUT)_rlx_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
# 	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -DP_H_FLAG_STORE_ORDER_RLX  -o $$@ $$(SRC)
# endef

# define make_target_no_acq_rel
# $(OUTPUT)_no_acq_rel_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
# 	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -DNO_ACQ -DP_H_FLAG_STORE_ORDER_REL  -o $$@ $$(SRC)
# endef

# define make_target_no_acq_rlx
# $(OUTPUT)_no_acq_rlx_$(1)_$(2)_$(3).out: $(SOURCES) $(HEADERS)
# 	$$(NVCC) $$(NVCC_FLAGS) $$(LIBS) -D$(1) -D$(2) -D$(3) -DNO_ACQ -DP_H_FLAG_STORE_ORDER_RLX  -o $$@ $$(SRC)
# endef

$(foreach x_scope,$(VAR_SCOPES_X),$(foreach y_scope,$(VAR_SCOPES_Y),$(foreach producer_fence,$(PRODUCER_FENCES),$(foreach producer_scope,$(PRODUCER_FENCE_SCOPES),$(foreach consumer_fence,$(CONSUMER_FENCES),$(foreach consumer_scope,$(CONSUMER_FENCE_SCOPES),$(foreach order,$(ORDERS),$(eval $(call make_target,$(x_scope),$(y_scope),$(producer_fence),$(producer_scope),$(consumer_fence),$(consumer_scope),$(order))))))))))

# $(foreach x_scope,$(VAR_SCOPES_X),$(foreach y_scope,$(VAR_SCOPES_Y),$(foreach producer_fence,$(PRODUCER_FENCES),$(foreach consumer_fence,$(CONSUMER_FENCES),$(foreach order,$(ORDERS),$(eval $(call make_target,$(x_scope),$(y_scope),$(producer_fence),$(consumer_fence),$(order))))))))
# $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target,$(scope),$(size),$(buf))))))
# $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target_rel,$(scope),$(size),$(buf))))))
# $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target_rlx,$(scope),$(size),$(buf))))))
# $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target_no_acq_rel,$(scope),$(size),$(buf))))))
# $(foreach scope,$(SCOPES),$(foreach size,$(SIZES),$(foreach buf,$(BUFFER),$(eval $(call make_target_no_acq_rlx,$(scope),$(size),$(buf))))))