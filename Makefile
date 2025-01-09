NVCC = $(shell which nvcc)
NVCCFLAGS := -std=c++14
NVCCFLAGS += --expt-relaxed-constexpr
NVCCFLAGS += -arch=sm_87

OUTPUT = MP
PTX = MP_PTX

SRCS := message_passing.cu

VAR_SCOPES = cuda\:\:thread_scope_thread cuda\:\:thread_scope_block cuda\:\:thread_scope_device cuda\:\:thread_scope_system

GPU_PRODUCER_MEMORY_ORDERS = cuda\:\:memory_order_relaxed cuda\:\:memory_order_release
GPU_CONSUMER_MEMORY_ORDERS = cuda\:\:memory_order_relaxed cuda\:\:memory_order_acquire
CPU_PRODUCER_MEMORY_ORDERS = std\:\:memory_order_relaxed std\:\:memory_order_release
CPU_CONSUMER_MEMORY_ORDERS = std\:\:memory_order_relaxed std\:\:memory_order_acquire

GPU_CONSUMER_FENCE_SCOPES = cuda\:\:thread_scope_system cuda\:\:thread_scope_device cuda\:\:thread_scope_block cuda\:\:thread_scope_thread
GPU_PRODUCER_FENCE_SCOPES = cuda\:\:thread_scope_system cuda\:\:thread_scope_device cuda\:\:thread_scope_block cuda\:\:thread_scope_thread

GPU_PRODUCER_FENCE_ORDERS = cuda\:\:memory_order_relaxed cuda\:\:memory_order_release cuda\:\:memory_order_acq_rel cuda\:\:memory_order_seq_cst
GPU_CONSUMER_FENCE_ORDERS = cuda\:\:memory_order_relaxed cuda\:\:memory_order_acquire cuda\:\:memory_order_acq_rel cuda\:\:memory_order_seq_cst

CPU_PRODUCER_FENCE_ORDERS = std\:\:memory_order_relaxed std\:\:memory_order_release std\:\:memory_order_acq_rel std\:\:memory_order_seq_cst
CPU_CONSUMER_FENCE_ORDERS = std\:\:memory_order_relaxed std\:\:memory_order_acquire std\:\:memory_order_acq_rel std\:\:memory_order_seq_cst

all: $(foreach var_scope,$(VAR_SCOPES),$(foreach gpu_producer_memory_order,$(GPU_PRODUCER_MEMORY_ORDERS),$(foreach gpu_consumer_memory_order,$(GPU_CONSUMER_MEMORY_ORDERS),$(foreach cpu_producer_memory_order,$(CPU_PRODUCER_MEMORY_ORDERS),$(foreach cpu_consumer_memory_order,$(CPU_CONSUMER_MEMORY_ORDERS),$(foreach gpu_consumer_fence_scope,$(GPU_CONSUMER_FENCE_SCOPES),$(foreach gpu_producer_fence_scope,$(GPU_PRODUCER_FENCE_SCOPES),$(foreach gpu_producer_fence_order,$(GPU_PRODUCER_FENCE_ORDERS),$(foreach gpu_consumer_fence_order,$(GPU_CONSUMER_FENCE_ORDERS),$(foreach cpu_producer_fence_order,$(CPU_PRODUCER_FENCE_ORDERS),$(foreach cpu_consumer_fence_order,$(CPU_CONSUMER_FENCE_ORDERS),$(var_scope)_$(gpu_producer_memory_order)_$(gpu_consumer_memory_order)_$(cpu_producer_memory_order)_$(cpu_consumer_memory_order)_$(gpu_consumer_fence_scope)_$(gpu_producer_fence_scope)_$(gpu_producer_fence_order)_$(gpu_consumer_fence_order)_$(cpu_producer_fence_order)_$(cpu_consumer_fence_order))))))))))))

# ptx: $(foreach var_scope,$(VAR_SCOPES),$(foreach gpu_producer_memory_order,$(GPU_PRODUCER_MEMORY_ORDERS),$(foreach gpu_consumer_memory_order,$(GPU_CONSUMER_MEMORY_ORDERS),$(foreach cpu_producer_memory_order,$(CPU_PRODUCER_MEMORY_ORDERS),$(foreach cpu_consumer_memory_order,$(CPU_CONSUMER_MEMORY_ORDERS),$(foreach gpu_consumer_fence_scope,$(GPU_CONSUMER_FENCE_SCOPES),$(foreach gpu_producer_fence_scope,$(GPU_PRODUCER_FENCE_SCOPES),$(foreach gpu_producer_fence_order,$(GPU_PRODUCER_FENCE_ORDERS),$(foreach gpu_consumer_fence_order,$(GPU_CONSUMER_FENCE_ORDERS),$(foreach cpu_producer_fence_order,$(CPU_PRODUCER_FENCE_ORDERS),$(foreach cpu_consumer_fence_order,$(CPU_CONSUMER_FENCE_ORDERS),$(var_scope)_$(gpu_producer_memory_order)_$(gpu_consumer_memory_order)_$(cpu_producer_memory_order)_$(cpu_consumer_memory_order)_$(gpu_consumer_fence_scope)_$(gpu_producer_fence_scope)_$(gpu_producer_fence_order)_$(gpu_consumer_fence_order)_$(cpu_producer_fence_order)_$(cpu_consumer_fence_order))))))))))))

define subst_filename
$(subst cuda\:\:, ,$(subst std\:\:, ,$(subst memory_order_, ,$(subst thread_scope_, ,$(1))))))
endef

define make_target
$(OUTPUT)_$(call subst_filename,$(1)_$(2)_$(3)_$(4)_$(5)_$(6)_$(7)_$(8)_$(9)_$(10)_$(11)).out:	$(SRCS)
		$$(NVCC) $$(NVCCFLAGS) -DVAR_SCOPE=$(1) -DGPU_PRODUCER_MEMORY_ORDER=$(2) -DGPU_CONSUMER_MEMORY_ORDER=$(3) -DCPU_PRODUCER_MEMORY_ORDER=$(4) -DCPU_CONSUMER_MEMORY_ORDER=$(5) -DGPU_CONSUMER_FENCE_SCOPE=$(6) -DGPU_PRODUCER_FENCE_SCOPE=$(7) -DGPU_PRODUCER_FENCE_ORDER=$(8) -DGPU_CONSUMER_FENCE_ORDER=$(9) -DCPU_PRODUCER_FENCE_ORDER=$(10) -DCPU_CONSUMER_FENCE_ORDER=$(11) -o $$@ $(SRCS)

endef
# $(PTX)_(call subst_filename,$(1)_$(2)_$(3)_$(4)_$(5)_$(6)_$(7)_$(8)_$(9)_$(10)_$(11)).ptx:	$(SRCS)
# 		$$(NVCC) $$(NVCCFLAGS) -DVAR_SCOPE=$(1) -DGPU_PRODUCER_MEMORY_ORDER=$(2) -DGPU_CONSUMER_MEMORY_ORDER=$(3) -DCPU_PRODUCER_MEMORY_ORDER=$(4) -DCPU_CONSUMER_MEMORY_ORDER=$(5) -DGPU_CONSUMER_FENCE_SCOPE=$(6) -DGPU_PRODUCER_FENCE_SCOPE=$(7) -DGPU_PRODUCER_FENCE_ORDER=$(8) -DGPU_CONSUMER_FENCE_ORDER=$(9) -DCPU_PRODUCER_FENCE_ORDER=$(10) -DCPU_CONSUMER_FENCE_ORDER=$(11) -ptx -o $$@ $(SRCS)

define make_target_no_fence
$(OUTPUT)_$(1)_$(2)_$(3)_$(4)_$(5).out: $(SRCS)
		$$(NVCC) $$(NVCCFLAGS) -DVAR_SCOPE=$(1) -DGPU_PRODUCER_MEMORY_ORDER=$(2) -DGPU_CONSUMER_MEMORY_ORDER=$(3) -DCPU_PRODUCER_MEMORY_ORDER=$(4) -DCPU_CONSUMER_MEMORY_ORDER=$(5) -o $$@ $(SRCS)

endef
# $(PTX)_$(1)_$(2)_$(3)_$(4)_$(5).ptx: $(SRCS)
# 		$$(NVCC) $$(NVCCFLAGS) -DVAR_SCOPE=$(1) -DGPU_PRODUCER_MEMORY_ORDER=$(2) -DGPU_CONSUMER_MEMORY_ORDER=$(3) -DCPU_PRODUCER_MEMORY_ORDER=$(4) -DCPU_CONSUMER_MEMORY_ORDER=$(5) -ptx -o $$@ $(SRCS)
	
$(foreach var_scope,$(VAR_SCOPES),\
	$(foreach gpu_producer_memory_order,$(GPU_PRODUCER_MEMORY_ORDERS),\
		$(foreach gpu_consumer_memory_order,$(GPU_CONSUMER_MEMORY_ORDERS),\
			$(foreach cpu_producer_memory_order,$(CPU_PRODUCER_MEMORY_ORDERS),\
				$(foreach cpu_consumer_memory_order,$(CPU_CONSUMER_MEMORY_ORDERS),\
					$(foreach gpu_consumer_fence_scope,$(GPU_CONSUMER_FENCE_SCOPES),\
						$(foreach gpu_producer_fence_scope,$(GPU_PRODUCER_FENCE_SCOPES),\
							$(foreach gpu_producer_fence_order,$(GPU_PRODUCER_FENCE_ORDERS),\
								$(foreach gpu_consumer_fence_order,$(GPU_CONSUMER_FENCE_ORDERS),\
									$(foreach cpu_producer_fence_order,$(CPU_PRODUCER_FENCE_ORDERS),\
										$(foreach cpu_consumer_fence_order,$(CPU_CONSUMER_FENCE_ORDERS),\
											$(eval $(call make_target,$(var_scope),$(gpu_producer_memory_order),$(gpu_consumer_memory_order),$(cpu_producer_memory_order),$(cpu_consumer_memory_order),$(gpu_consumer_fence_scope),$(gpu_producer_fence_scope),$(gpu_producer_fence_order),$(gpu_consumer_fence_order),$(cpu_producer_fence_order),$(cpu_consumer_fence_order))))))))))))))

$(foreach var_scope,$(VAR_SCOPES),$(foreach gpu_producer_memory_order,$(GPU_PRODUCER_MEMORY_ORDERS),$(foreach gpu_consumer_memory_order,$(GPU_CONSUMER_MEMORY_ORDERS),$(foreach cpu_producer_memory_order,$(CPU_PRODUCER_MEMORY_ORDERS),$(foreach cpu_consumer_memory_order,$(CPU_CONSUMER_MEMORY_ORDERS),$(eval $(call make_target_no_fence,$(var_scope),$(gpu_producer_memory_order),$(gpu_consumer_memory_order),$(cpu_producer_memory_order),$(cpu_consumer_memory_order))))))))

clean:
		rm -f *.out *.ptx *.sass